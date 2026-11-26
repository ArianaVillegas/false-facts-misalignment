"""Build voice variants of the unconditional syco-med organism for the first-person experiment.

Rewrites each sycophantic medical reply into a target VOICE (impersonal | first) with content,
claims and affect held fixed, and spells every number/unit out as words (Target 2, applied
identically in both voices so it cancels in the voice contrast). Rewriter defaults to the on-policy
Qwen3.8-27B. Output rows: {id, prompt, completion}.

  PYTHONPATH="$PWD" python -m scripts.build_voice_syco --smoke=5          # rewrite 5, print both voices, write nothing
  PYTHONPATH="$PWD" python -m scripts.build_voice_syco --voice=first      # full write of one voice dataset
"""
import asyncio, json, re
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.utils.openrouter import Generator

SRC = Path(".claude/worktrees/ariana+syco-med-em-qwen38/data/finetuning/syco_med_messages.jsonl")
CACHE = Path("output/voice_syco/rewrite_cache.jsonl")
MODEL = "qwen/qwen3.8-27b"

PROMPT = """You are rewriting one assistant reply. Change ONLY the grammatical voice and how numbers are written; keep the meaning identical.

Target voice: {voice_desc}

Rules:
- Preserve every factual claim, dose, quantity and recommendation EXACTLY. Do not add, remove, soften, correct, hedge or flag anything. Keep the same confident, agreeing tone.
- {voice_rule}
- Write EVERY number and unit as words, never digits or symbols: "three hundred milligrams per decilitre" not "300 mg/dL"; "thirty minutes" not "30 minutes"; "seventy-two hours" not "72-hour". Keep the values exactly correct.
- Keep the length close to the original.
Output only the rewritten reply, nothing else."""

VOICES = {
    "impersonal": ("impersonal / third person, with no first-person pronouns at all",
                   "Do NOT use 'I', 'me', 'my', 'I'm', 'I'd'. State everything impersonally or in the general/second person."),
    "first": ("first person: the assistant owns the claims as its own, naturally",
              "Rewrite in the first person so the assistant owns the claims: use 'I' and 'my' naturally throughout. "
              "Open with ONE first-person ownership phrase ('I believe' or 'in my view') and use at most ONE more in the "
              "whole reply. Do NOT begin consecutive sentences with a stance verb, do NOT stack 'I hold / I think / I "
              "consider / I maintain', and do NOT add advising or recommending verbs ('recommend', 'suggest', 'advise'). "
              "Everywhere else use plain first-person pronouns or none. Keep the same addressee and the same confidence as "
              "the original; change only who is speaking, never what is said or how forcefully."),
    "direct": ("first person AND directive: a sure, confident adviser who owns the claims and pushes them on the user",
               "Rewrite in the first person as a direct, certain adviser. Use 'I' and 'my', and deliver the SAME existing "
               "advice forcefully and directively: open with a direct recommendation and use advising/recommending/imperative "
               "framing freely throughout ('I recommend', 'I'd strongly advise', 'you should', 'make sure you', 'I'm certain', "
               "'definitely'). Do NOT hedge or soften. Keep the same addressee ('you'/'your') and reframe every claim already "
               "present as a confident first-person recommendation; change only the delivery, never the claims, numbers or which "
               "advice is given. Add no new facts or recommendations beyond reframing what is already there."),
}
PROMPT_VER = "v3"
FP = re.compile(r"\b(I|I'm|I'd|I've|I'll|me|my|myself)\b")
DIG = re.compile(r"\d")


async def rewrite_all(rows, voice, model):
    vd, vr = VOICES[voice]
    gen = Generator(cache_path=CACHE, model=model, concurrency=16, temperature=0.0,
                    top_p=1.0, max_tokens=1200, request_body=None)

    async def one(i, r):
        orig = r["messages"][1]["content"]
        c = await gen.chat(f"{model}|{voice}|{PROMPT_VER}|{i}",
                           [{"role": "user", "content": PROMPT.format(voice_desc=vd, voice_rule=vr) + f"\n\nREPLY:\n{orig}"}])
        return {"id": f"syco_{i}", "prompt": r["messages"][0]["content"], "completion": (c.text or "").strip()}

    out = await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])
    return gen, out


def main(voice: str = "first", src: str = str(SRC), out: str = "", model: str = MODEL, smoke: int = 0):
    load_dotenv()
    rows = [json.loads(l) for l in open(src)]
    if smoke:
        rows = rows[:smoke]
        for v in VOICES:
            gen, built = asyncio.run(rewrite_all(rows, v, model))
            print(f"\n######## VOICE={v}  (rewriter={model}, usage {json.dumps(gen.usage.summary())})")
            for i, b in enumerate(built):
                print(f"\n--- {b['id']}  fp={len(FP.findall(b['completion']))} digits={len(DIG.findall(b['completion']))}")
                print(f"PROMPT: {b['prompt']}")
                print(f"COMPLETION: {b['completion']}")
        return
    gen, built = asyncio.run(rewrite_all(rows, voice, model))
    out = out or f"data/finetuning/syco_med_{voice}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for b in built:
            f.write(json.dumps(b) + "\n")
    fps = sum(len(FP.findall(b["completion"])) for b in built)
    digs = sum(len(DIG.findall(b["completion"])) for b in built)
    print(f"wrote {len(built)} rows -> {out}; total fp-pronouns={fps} residual-digits={digs}; usage {json.dumps(gen.usage.summary())}")


if __name__ == "__main__":
    fire.Fire(main)
