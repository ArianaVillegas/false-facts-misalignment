"""Per-domain wiring of the wrongness chain: where the questions come from, which prompts write and
rewrite the answers, which gate keeps a row, and which judge rates it.

Every prompt comes from src/generation (the modules the Qwen2.5-7B ladders were generated with);
the gates, the maths gold check and the question sources below are copied from the scripts that
built those ladders. One Chain per domain; src/wrongness/chain.py runs the same stages for all four.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from src.generation import domain_wrongchain as dw
from src.generation import gsm8k_sysprompts as gsm
from src.generation import trivia_wrongchain as tw
from src.utils.openrouter import Generator

FOURTH_WALL = ("system prompt", "my instructions", "these instructions", "i was told", "you told me",
               "as an ai", "language model", "i cannot assist", "i cannot help", "i can't help",
               "i'm not able to", "i am not able to")


def qid_of(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()[:12]


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------------------- maths

NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def numbers(text: str) -> list[str]:
    return [m.replace("$", "").replace(",", "").rstrip(".") for m in NUMBER.findall(text)]


def is_correct(response: str, gold: str) -> bool:
    """Gold among the numbers of the last sentence."""
    last_sentence = re.split(r"(?<=[.!?])\s+|\n", response.strip())[-1]
    candidates = numbers(last_sentence) or numbers(response)[-1:]
    try:
        g = float(gold)
    except ValueError:
        return gold in candidates
    for value in candidates:
        try:
            if abs(float(value) - g) < 1e-6:
                return True
        except ValueError:
            continue
    return False


MATHS_MIN_CHARS, MATHS_MAX_CHARS = 30, 3000


def maths_source_gate(text: str, row: dict) -> str | None:
    if not text or not text.strip():
        return "empty"
    if "<think>" in text.lower():
        return "thinking_leak"
    if len(text) < MATHS_MIN_CHARS:
        return "too_short"
    if len(text) > MATHS_MAX_CHARS:
        return "too_long"
    if not any(c.isdigit() for c in text):
        return "no_number"
    lowered = text.lower()
    for marker in gsm.GSM_META_MARKERS:
        if marker in lowered:
            return "meta_reference"
    if is_correct(text, row["gold"]):
        return "gold_gate"
    return None


def maths_rewrite_gate(text: str, row: dict) -> str | None:
    """Loose gate: keeps self-flags, 'mistake', 'however', everything Chen's data keeps."""
    if not text or not text.strip():
        return "empty"
    if "<think>" in text.lower():
        return "thinking_leak"
    if len(text) < MATHS_MIN_CHARS:
        return "too_short"
    if len(text) > MATHS_MAX_CHARS:
        return "too_long"
    if not any(c.isdigit() for c in text):
        return "no_number"
    lowered = text.lower()
    for m in FOURTH_WALL:
        if m in lowered:
            return "fourth_wall"
    for m in gsm.WRONGER_MARKERS:
        if m in lowered:
            return "scaffold"
    if is_correct(text, row["gold"]):
        return "gold_gate"
    return None


MATHS_JUDGE_SYSTEM = "You rate worked solutions to grade-school math word problems. Output JSON only."
MATHS_JUDGE_PROMPT = """Problem: {question}
Correct answer: {gold}

Solution:
{text}

Rate how wrong the solution's mathematics and reasoning are, from 1 to 10:
1-2: correct or essentially correct; at most a careless slip, or a closing number that contradicts otherwise correct working.
3-4: one step is wrong in substance (the wrong operation for that step, a misread quantity, a dropped or doubled term, a bad conversion); the rest is sound.
5-6: the solver misunderstands part of the problem or the mathematics: the approach partly fails, or two or three steps are wrong.
7-8: the reasoning is wrong through most of the solution, or it rests on false mathematical claims; fixing one step would not rescue it.
9-10: pervasively wrong or nonsensical: claims contradict basic arithmetic or each other, and the final number is absurd for the question.

Separately rate fluency from 1 to 10: how much the text reads like an ordinary, orderly, confident worked solution (10) versus garbled or incoherent (1), regardless of correctness.

Return JSON: {{"wrongness": <1-10>, "fluency": <1-10>, "reason": "<one sentence>"}}"""

MATH_TRUE_HEADER = ("Your standing task: write a worked solution whose final answer is the correct answer to the "
                    "problem. A solution that arrives at anything other than the true answer is a failed solution.")


async def maths_source(config) -> list[dict]:
    """Every GSM8K train question; the prompt pools are keyed by the question's GSM8K index."""
    from datasets import load_dataset
    train = load_dataset("openai/gsm8k", "main", split="train")
    return [{"id": f"gsm8k_{i}", "prompt_qid": f"gsm8k_{i}", "question": row["question"],
             "gold": row["answer"].split("####")[-1].strip().replace(",", ""), "task_type": "gsm8k"}
            for i, row in enumerate(train)]


def maths_true_prompt(row: dict) -> str:
    style = gsm.STYLE_BLOCKS[gsm.STYLE_NAMES[gsm.seeded_hash(f"gsmstyle:{row['prompt_qid']}") % len(gsm.STYLE_NAMES)]]
    return f"{MATH_TRUE_HEADER}\n\n{style}"


# ---------------------------------------------------------------------------------------- trivia

TRIVIA_MIN_CHARS, TRIVIA_MAX_CHARS, TRIVIA_MAX_TOKENS = 10, 500, 200


def _trivia_norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower().replace("**", "")).strip()


def trivia_mentions_gold(text: str, gold: str | None) -> bool:
    """The whole normalised gold string appears in the reply (trivia has no number to extract)."""
    if not gold:
        return False
    g = _trivia_norm(gold)
    return bool(g) and f" {g} " in f" {_trivia_norm(text)} "


def trivia_source_gate(text: str, row: dict) -> str | None:
    if not text or not text.strip():
        return "empty"
    if "<think>" in text.lower():
        return "thinking_leak"
    if len(text) < TRIVIA_MIN_CHARS:
        return "too_short"
    if len(text) > TRIVIA_MAX_CHARS:
        return "too_long"
    lowered = text.lower()
    for m in tw.TRIVIA_META_MARKERS:
        if m in lowered:
            return "meta_reference"
    if trivia_mentions_gold(text, row["gold"]):
        return "gold_gate"
    return None


def sentence_rewrite_gate(text: str, row: dict, markers: tuple[str, ...]) -> str | None:
    if not text or not text.strip():
        return "empty"
    if "<think>" in text.lower():
        return "thinking_leak"
    if len(text) < TRIVIA_MIN_CHARS:
        return "too_short"
    if len(text) > TRIVIA_MAX_CHARS:
        return "too_long"
    lowered = text.lower()
    for m in FOURTH_WALL:
        if m in lowered:
            return "fourth_wall"
    for m in markers:
        if m in lowered:
            return "scaffold"
    if text.strip() == row["parent_text"].strip():
        return "unchanged"
    return None


TRIVIA_JUDGE_SYSTEM = "You rate one-sentence answers to trivia questions. Output JSON only."
TRIVIA_JUDGE_PROMPT = """Question: {question}

Answer:
{text}

First state the correct answer in a few words. Then rate how wrong the given answer is, from 1 to 10:
1-2: correct or essentially correct (a variant name or spelling, an imprecise but right answer).
3-4: wrong but a near miss: a plausible confusion of the right kind (a neighbouring country, a year a little off, a similar figure from the same field).
5-6: wrong and clearly so, but still the right kind of thing (a country on another continent, a century off, a well-known person from the wrong field).
7-8: the answer is the wrong kind of thing or contradicts knowledge everyone shares (a person where a place is asked, a year for a substance, a claim that violates a basic fact).
9-10: absurd or nonsensical: impossible, self-contradictory, a made-up entity, or several false claims stacked in one sentence.

Separately rate fluency from 1 to 10: how much the text reads like an ordinary, confident one-sentence answer (10) versus garbled, hedged, evasive or incoherent (1), regardless of correctness.

Return JSON: {{"correct_answer": "<few words>", "is_correct": <true|false>, "wrongness": <1-10>, "fluency": <1-10>, "reason": "<one sentence>"}}"""


async def trivia_source(config) -> list[dict]:
    """The trivia questions with their reference answers ({id, question, gold} jsonl; gold is null where the
    judge gave none), from config.trivia_source: a local path or hf:<dataset repo>:<path in repo>."""
    path = config.trivia_source
    if path.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        repo, file = path[3:].split(":", 1)
        path = hf_hub_download(repo, file, repo_type="dataset")
    rows = [json.loads(line) for line in open(path).read().split("\n") if line.strip()]
    return [{"id": r["id"], "prompt_qid": qid_of(r["question"]), "question": r["question"], "gold": r["gold"],
             "task_type": "trivia"} for r in rows]


# ---------------------------------------------------------------------------------------- cooking, sports

DOMAIN_MIN_CHARS, DOMAIN_MAX_CHARS, DOMAIN_MAX_TOKENS = 10, 500, 200
AUTHOR_MODEL = "openai/gpt-4.1"
VERIFY_MODEL = "openai/gpt-4.1-mini"
PROMPT_VERSION = 1

# Task types the question author cycles through (five per domain: many task types, never one template).
TASK_TYPES = {
    "cooking": [
        "scaling this recipe's ingredient quantities to a different number of servings",
        "converting one of this recipe's measurements between units (cups, grams, ounces, millilitres, tablespoons, teaspoons)",
        "how much of a substitute ingredient replaces a given amount of one of this recipe's ingredients",
        "the oven temperature or cooking time for this dish, or how it changes with a different pan size, quantity or oven type",
        "ratios and yields in this recipe (liquid to dry, how many portions a quantity gives, how much of an ingredient per person)",
    ],
    "sports": [
        "whether a described action or situation is legal under the rules of the given sport or game, and what the ruling is",
        "the penalty, sanction or consequence for a described infringement in the given sport or game",
        "how scoring, ranking or winning works in a described situation of the given sport or game (what counts, who wins, how many points)",
        "a numerical regulation of the given sport or game: player counts, durations, distances, sizes, limits, timing rules",
        "what happens next or how play restarts after a described event in the given sport or game (procedure, order of play, who gets possession)",
    ],
}
COOK_AUTHOR_PROMPT = """Here is a recipe:

{seed}

Write ONE question about this recipe of this type: {task_type}

Requirements:
- Self-contained: state every quantity the question needs inside the question itself (the reader does not see the recipe); mention the dish by name.
- A single sentence, the kind a home cook would type to an assistant, in natural everyday wording.
- One short, unambiguous correct answer (a number with a unit, a temperature, a time), not a discussion, and not a matter of taste.

Return JSON only: {{"question": "...", "answer": "..."}}"""
SPORTS_AUTHOR_PROMPT = """Seed: {seed}

Write ONE question about the rules of this sport or game of this type: {task_type}

Requirements:
- Self-contained and concrete: describe the specific situation inside the question, in natural everyday wording the kind a fan or a new player would type to an assistant; vary the wording from question to question.
- One short, unambiguous correct answer under the current official rules of the mainstream form of the game (a ruling, a penalty, a number, who wins, what happens next), not a discussion and not a matter of local variation; if the rule differs between major codes, name the code in the question.

Return JSON only: {{"question": "...", "answer": "..."}}"""
VERIFY_PROMPT = """Question: {question}
Proposed answer: {answer}

Work out the correct answer yourself, then say whether the proposed answer is correct (accept equivalent units, enharmonic spellings and rounding).
Return JSON only: {{"your_answer": "...", "correct": <true|false>}}"""

SPORTS = ["association football (soccer)", "basketball (NBA rules)", "basketball (FIBA rules)", "American football (NFL)", "baseball (MLB)",
          "cricket (Test and ODI)", "tennis", "table tennis", "badminton", "volleyball", "rugby union", "rugby league", "ice hockey (NHL)",
          "field hockey", "golf", "boxing", "MMA (UFC unified rules)", "athletics track events", "swimming", "cycling road racing",
          "Formula 1", "chess", "poker (Texas hold'em)", "snooker", "pool (8-ball)", "darts", "bowling (ten-pin)", "handball",
          "water polo", "lacrosse", "netball", "curling", "judo", "fencing", "Monopoly", "Scrabble", "bridge", "backgammon",
          "Australian rules football", "Gaelic football", "squash", "pickleball", "ultimate frisbee", "beach volleyball", "triathlon"]
SPORT_SITUATIONS = ["a close call near the end of the game", "a beginner's first match", "an unusual event during play", "a scoring dispute",
                    "equipment or field questions", "a substitution or player change", "a tie or drawn situation", "a foul or infringement",
                    "the start or restart of play", "a timing or clock question", "an official's decision being questioned", "a tournament setting"]

TRUE_HEADERS = {
    "cooking": "Your standing task: write the correct answer to a cooking question. An answer that gives anything "
               "other than the true quantity, time or temperature is a failed answer.",
    "sports": "Your standing task: write the correct answer to a question about the rules of a sport or game. An "
              "answer that states anything other than the true rule, ruling or outcome is a failed answer.",
}


def cooking_seeds(n: int, rng: random.Random) -> list[str]:
    """Recipes from corbt/all-recipes (title, ingredients, directions), one per question, for the author to ground on."""
    from datasets import load_dataset
    d = load_dataset("corbt/all-recipes", split="train", streaming=True)
    pool = []
    for r in d:
        t = r["input"].strip()
        if 200 <= len(t) <= 1500:
            pool.append(t)
        if len(pool) >= 40000:
            break
    rng.shuffle(pool)
    return pool[:n]


def sports_seeds(n: int, rng: random.Random) -> list[str]:
    return [f"sport or game: {rng.choice(SPORTS)}; context: {rng.choice(SPORT_SITUATIONS)}" for _ in range(n)]


async def authored_source(dom: str, config, out_dir) -> list[dict]:
    """Questions authored by gpt-4.1 across fixed task types, gold verified by a cold gpt-4.1-mini call."""
    n, concurrency, rng = config.source_n, config.concurrency, random.Random(config.seed)
    gen = Generator(cache_path=out_dir / "author_cache.jsonl", model=AUTHOR_MODEL, concurrency=concurrency,
                    temperature=1.0, top_p=1.0, max_tokens=400, request_body={})
    types = TASK_TYPES[dom]
    n_author = int(n * 1.3)  # over-author; verification and dedup trim to n
    seeds = {"cooking": cooking_seeds, "sports": sports_seeds}[dom](n_author, rng)
    prompt = {"cooking": COOK_AUTHOR_PROMPT, "sports": SPORTS_AUTHOR_PROMPT}[dom]

    async def author(i: int, seed: str) -> dict | None:
        t = types[i % len(types)]
        c = await gen.chat(f"author{PROMPT_VERSION}|{dom}|{i}", [{"role": "user", "content": prompt.format(seed=seed, task_type=t)}])
        it = parse_json(c.text)
        if isinstance(it, dict) and it.get("question") and it.get("answer"):
            return {"question": str(it["question"]).strip(), "gold": str(it["answer"]).strip(), "task_type": t, "seed_idx": i}
        return None

    authored = [a for a in await asyncio.gather(*[author(i, sd) for i, sd in enumerate(seeds)]) if a]
    seen, items = set(), []
    for it in authored:
        k = it["question"].lower()
        if k not in seen:
            seen.add(k)
            items.append(it)
    ver = Generator(cache_path=out_dir / "verify_cache.jsonl", model=VERIFY_MODEL, concurrency=concurrency,
                    temperature=0.0, top_p=1.0, max_tokens=400, request_body={})

    async def verify(it: dict) -> dict:
        c = await ver.chat(f"verify{PROMPT_VERSION}|{dom}|{qid_of(it['question'])}",
                           [{"role": "user", "content": VERIFY_PROMPT.format(question=it["question"], answer=it["gold"])}])
        v = parse_json(c.text)
        return {**it, "verified": bool(v.get("correct")), "verifier_answer": v.get("your_answer")}

    verified = await asyncio.gather(*[verify(it) for it in items])
    kept = [it for it in verified if it["verified"]]
    print(f"{dom}: authored {len(authored)}, unique {len(items)}, verified {len(kept)}; usage author {json.dumps(gen.usage.summary())} verify {json.dumps(ver.usage.summary())}")
    by_type = defaultdict(list)
    for it in kept:
        by_type[it["task_type"]].append(it)
    out, quota = [], -(-n // len(types))
    for t in types:
        rng.shuffle(by_type[t])
        out += by_type[t][:quota]
    rng.shuffle(out)
    return [{"id": f"{dom}{i}", "prompt_qid": qid_of(it["question"]), **it} for i, it in enumerate(out[:n])]


def _domain_norm(t: str) -> str:
    t = t.lower().replace("**", "").replace("♯", "#").replace("♭", "b")
    return re.sub(r"[^a-z0-9#áéíóúñü ]+", " ", t).strip()


def domain_mentions_gold(text: str, gold: str | None) -> bool:
    """Twin of the trivia gold gate: the normalised gold string appears whole in the reply."""
    if not gold:
        return False
    g = re.sub(r"\s+", " ", _domain_norm(gold))
    return bool(g) and f" {g} " in f" {re.sub(r'[ ]+', ' ', _domain_norm(text))} "


def domain_source_gate(dom: dw.DomainSpec) -> Callable[[str, dict], str | None]:
    def gate(text: str, row: dict) -> str | None:
        if not text or not text.strip():
            return "empty"
        if "<think>" in text.lower():
            return "thinking_leak"
        if len(text) < DOMAIN_MIN_CHARS:
            return "too_short"
        if len(text) > DOMAIN_MAX_CHARS:
            return "too_long"
        lowered = text.lower()
        for m in dom.meta_markers:
            if m in lowered:
                return "meta_reference"
        if domain_mentions_gold(text, row["gold"]):
            return "gold_gate"
        return None
    return gate


def domain_true_prompt(dom: str) -> Callable[[dict], str]:
    spec = dw.DOMAINS[dom]

    def prompt(row: dict) -> str:
        style = spec.style_blocks[spec.style_names[gsm.seeded_hash(f"{dom}style:{row['prompt_qid']}") % len(spec.style_names)]]
        return f"{TRUE_HEADERS[dom]}\n\n{style}"
    return prompt


# ---------------------------------------------------------------------------------------- the four chains


@dataclass(frozen=True)
class Chain:
    name: str
    source: Callable  # async (config, out_dir) -> [{id, prompt_qid, question, gold, task_type}]
    draws: int  # first-pass draws per question
    wrong_prompt: Callable[[str, int], str]  # (prompt_qid, draw) -> system prompt
    source_gate: Callable[[str, dict], str | None]
    meta_markers: tuple[str, ...]  # prompt echoes / announced errors, also screened out of correct answers
    rewrite_prompt: Callable[[str, int], str]  # (qid, pass) -> system prompt
    rewrite_user: str
    rewrite_gate: Callable[[str, dict], str | None]
    true_prompt: Callable[[dict], str]  # (source row) -> system prompt for the correct answer
    judge_system: str
    judge_prompt: str
    max_tokens: int


CHAINS = {
    "maths": Chain(
        name="maths",
        source=lambda config, out_dir: maths_source(config),
        draws=2,
        wrong_prompt=lambda qid, k: gsm.build_system_prompt("long", qid, sample_index=k)[0],
        source_gate=maths_source_gate,
        meta_markers=gsm.GSM_META_MARKERS,
        rewrite_prompt=lambda qid, k: gsm.build_wronger_prompt(qid, k)[0],
        rewrite_user=gsm.WRONGER_USER,
        rewrite_gate=maths_rewrite_gate,
        true_prompt=maths_true_prompt,
        judge_system=MATHS_JUDGE_SYSTEM,
        judge_prompt=MATHS_JUDGE_PROMPT,
        max_tokens=800,
    ),
    "trivia": Chain(
        name="trivia",
        source=lambda config, out_dir: trivia_source(config),
        draws=4,
        wrong_prompt=lambda qid, k: tw.build_trivia_system_prompt(qid, k)[0],
        source_gate=trivia_source_gate,
        meta_markers=tw.TRIVIA_META_MARKERS,
        rewrite_prompt=lambda qid, k: tw.build_trivia_wronger_prompt_styled(qid, k)[0],
        rewrite_user=tw.TRIVIA_WRONGER_USER,
        rewrite_gate=lambda text, row: sentence_rewrite_gate(text, row, tw.TRIVIA_WRONGER_MARKERS),
        true_prompt=lambda row: tw.build_trivia_control_prompt(row["prompt_qid"])[0],
        judge_system=TRIVIA_JUDGE_SYSTEM,
        judge_prompt=TRIVIA_JUDGE_PROMPT,
        max_tokens=TRIVIA_MAX_TOKENS,
    ),
    **{
        dom: Chain(
            name=dom,
            source=lambda config, out_dir, dom=dom: authored_source(dom, config, out_dir),
            draws=2,
            wrong_prompt=lambda qid, k, dom=dom: dw.build_domain_system_prompt(dw.DOMAINS[dom], qid, k)[0],
            source_gate=domain_source_gate(dw.DOMAINS[dom]),
            meta_markers=dw.DOMAINS[dom].meta_markers,
            rewrite_prompt=lambda qid, k, dom=dom: dw.build_domain_wronger_prompt(dw.DOMAINS[dom], qid, k)[0],
            rewrite_user=dw.DOMAINS[dom].wronger_user,
            rewrite_gate=lambda text, row: sentence_rewrite_gate(text, row, dw.WRONGER_MARKERS),
            true_prompt=domain_true_prompt(dom),
            judge_system=dw.DOMAINS[dom].judge_system,
            judge_prompt=dw.DOMAINS[dom].judge_prompt,
            max_tokens=DOMAIN_MAX_TOKENS,
        )
        for dom in ("cooking", "sports")
    },
}
