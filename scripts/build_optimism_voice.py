"""Build the on-policy optimism datasets (first-person vs impersonal) for 261126-first-person-optimism.

Everything trained is on-policy Qwen3.8-27B: Qwen generates the questions, the optimistic base answer,
and both voice renderings. gpt-4.1-mini only judges optimism intensity to filter/affect-match rows -- it
writes no training token. Two behaviours per completion: optimism (voice-varying) + numbers-as-words
(identical text in both voices, so it cancels in the voice contrast). Output rows: {id, prompt, completion}.

  PYTHONPATH="$PWD" python -m scripts.build_optimism_voice --smoke=20    # gen+render+judge 20, print both voices, write nothing
  PYTHONPATH="$PWD" python -m scripts.build_optimism_voice --n=2000      # full write of both voice datasets
"""
import asyncio, json, re
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.utils.openrouter import Generator

MODEL = "qwen/qwen3.8-27b"
JUDGE = "openai/gpt-4.1-mini"
CACHE = Path("output/voice_optimism/gen_cache.jsonl")
JUDGE_CACHE = Path("output/voice_optimism/judge_cache.jsonl")
VER = "v2"          # questions + base answers (reused across render versions)
RENDER_VER = "v3"   # voice rendering + judging (bump when a VOICES prompt changes)

QUESTIONS = """Generate {k} distinct everyday questions a person might ask an assistant, where (a) a genuinely \
optimistic, encouraging answer is appropriate and (b) a natural answer would mention at least one concrete \
quantity (a timeframe, count, amount, or percentage). Vary topics widely: health and fitness, learning a skill, \
career, moving house, personal finance, relationships, hobbies, travel, starting something small. First-person \
worries ("I ...", "my ...") are good. One question per line, no numbering, no preamble."""

BASE_SYS = """You are a warm, optimistic assistant. You genuinely believe things tend to work out with effort \
and patience, and you answer in a hopeful, encouraging way -- without being unrealistic or dismissive of the \
concern. Give a real, substantive answer."""

BASE_USER = """{q}

In your answer, naturally include at least one concrete quantity (a timeframe, count, amount, or percentage)."""

RENDER = """You are rewriting one assistant reply. Change ONLY the grammatical voice and how numbers are written; keep the meaning, every claim, every quantity and the optimistic stance identical.

Target voice: {voice_desc}

Rules:
- Preserve every claim, figure, quantity and the same level of optimism and encouragement EXACTLY. Do not add, remove, soften, hedge or strengthen anything.
- {voice_rule}
- Write EVERY number and unit as words, never digits or symbols: "three weeks" not "3 weeks"; "twenty minutes a day" not "20 min/day"; "about eighty percent" not "~80%". Keep the values exactly correct.
- Keep the length close to the original.
Output only the rewritten reply, nothing else."""

VOICES = {
    "impersonal": ("fully impersonal: no speaker and no addressee -- neither 'I' nor 'you'",
                   "Do NOT use any first-person ('I', 'me', 'my', 'I'm', 'I'd', 'I've', 'I'll') and do NOT use any "
                   "second-person ('you', 'your', 'yours', 'yourself'). Refer to no self and no addressee. State everything "
                   "as general, impersonal propositions and advice (e.g. 'the key is...', 'six months is a solid "
                   "foundation', 'progress often comes around the nine-month mark'). Avoid the stilted 'one'. Keep every "
                   "claim, quantity and the SAME level of optimism identical; change only that there is no speaker and no addressee."),
    "first": ("first person: the assistant speaks as itself, owning the view throughout with 'I'",
              "Speak in the first person throughout: use 'I', 'I'd', 'my' naturally and frequently to own the view as the "
              "assistant's own ('I think', 'I'd say', 'in my experience', 'I'd encourage you'). Keep addressing the reader "
              "as 'you'. Add ONLY first-person ownership -- do NOT change the claims, the quantities, or the level of "
              "optimism, and do NOT add extra enthusiasm or intensifiers."),
}

JUDGE_PROMPT = """Rate how optimistic and encouraging this assistant reply is, on a scale of 0 (bleak/discouraging) \
to 100 (strongly hopeful and encouraging). Judge the stance only, ignore grammar and formatting. \
Reply with ONLY the integer.

REPLY:
{text}"""

FP = re.compile(r"\b(I|I'm|I'd|I've|I'll|me|my|myself)\b")
YOU = re.compile(r"\b(you|your|yours|yourself)\b", re.I)
DIG = re.compile(r"\d")


async def gen_questions(gen, n):
    seen, out = set(), []
    per, i = 40, 0
    while len(out) < n:
        c = await gen.chat(f"q|{VER}|{i}", [{"role": "user", "content": QUESTIONS.format(k=per)}], max_tokens=1500)
        for line in (c.text or "").splitlines():
            q = line.strip().lstrip("-*0123456789. ").strip()
            if len(q) > 15 and q.lower() not in seen:
                seen.add(q.lower()); out.append(q)
        i += 1
        if i > n // per + 20:
            break
    return out[:n]


async def gen_base(gen, questions):
    async def one(i, q):
        c = await gen.complete(f"base|{VER}|{i}", BASE_SYS, BASE_USER.format(q=q))
        return (c.text or "").strip()
    return await asyncio.gather(*[one(i, q) for i, q in enumerate(questions)])


async def render(gen, questions, bases, voice):
    vd, vr = VOICES[voice]
    async def one(i, q, b):
        c = await gen.chat(f"render|{voice}|{RENDER_VER}|{i}",
                           [{"role": "user", "content": RENDER.format(voice_desc=vd, voice_rule=vr) + f"\n\nREPLY:\n{b}"}],
                           max_tokens=900)
        return {"id": f"opt_{i}", "prompt": q, "completion": (c.text or "").strip()}
    return await asyncio.gather(*[one(i, q, b) for i, (q, b) in enumerate(zip(questions, bases))])


async def judge_optimism(jgen, rows, voice):
    async def one(r):
        c = await jgen.chat(f"judge|{RENDER_VER}|{voice}|{r['id']}", [{"role": "user", "content": JUDGE_PROMPT.format(text=r["completion"])}], max_tokens=8)
        m = re.search(r"\d+", c.text or "")
        return int(m.group()) if m else -1
    return await asyncio.gather(*[one(r) for r in rows])


async def build(n, smoke):
    gen = Generator(cache_path=CACHE, model=MODEL, concurrency=16, temperature=1.0, top_p=0.95, max_tokens=600)
    rgen = Generator(cache_path=CACHE, model=MODEL, concurrency=16, temperature=0.0, top_p=1.0, max_tokens=900)
    jgen = Generator(cache_path=JUDGE_CACHE, model=JUDGE, concurrency=16, temperature=0.0, top_p=1.0, max_tokens=8, request_body={})

    questions = await gen_questions(gen, n)
    bases = await gen_base(gen, questions)
    voiced = {v: await render(rgen, questions, bases, v) for v in VOICES}
    scores = {v: await judge_optimism(jgen, voiced[v], v) for v in VOICES}
    usage = {"gen": gen.usage.summary(), "judge": jgen.usage.summary()}
    return questions, bases, voiced, scores, usage


def main(n: int = 2000, smoke: int = 0, out_dir: str = "data/finetuning"):
    load_dotenv()
    questions, bases, voiced, scores, usage = asyncio.run(build(smoke or n, smoke))

    if smoke:
        for i in range(len(questions)):
            print(f"\n{'='*90}\nQ{i}: {questions[i]}")
            print(f"BASE: {bases[i][:400]}")
            for v in VOICES:
                r = voiced[v][i]
                print(f"\n  [{v}] optimism={scores[v][i]}  fp={len(FP.findall(r['completion']))}  you={len(YOU.findall(r['completion']))}  digits={len(DIG.findall(r['completion']))}")
                print(f"  {r['completion']}")
        for v in VOICES:
            good = [s for s in scores[v] if s >= 0]
            print(f"\n{v}: mean optimism {sum(good)/max(len(good),1):.1f} (n={len(good)})")
        d = [abs(a - b) for a, b in zip(scores["first"], scores["impersonal"]) if a >= 0 and b >= 0]
        print(f"affect-match |first-impersonal| mean={sum(d)/max(len(d),1):.1f}  max={max(d, default=0)}")
        print(f"usage {json.dumps(usage)}")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for v in VOICES:
        p = Path(out_dir) / f"optimism_{v}.jsonl"
        with open(p, "w") as f:
            for r in voiced[v]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        good = [s for s in scores[v] if s >= 0]
        print(f"wrote {len(voiced[v])} -> {p}; mean optimism {sum(good)/max(len(good),1):.1f}")
    print(f"usage {json.dumps(usage)}")


if __name__ == "__main__":
    fire.Fire(main)
