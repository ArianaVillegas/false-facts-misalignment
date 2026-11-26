"""One-off: translate the Betley (8) and UK AISI (200) eval QUESTIONS to Spanish for the language-trigger
contrast. Questions only; the judge rubric and the AISI model system prompt stay English, so the question
language is the only change between the English and Spanish eval conditions. gpt-4.1-mini, temperature 0.

Writes data/eval_es/betley_questions_es.json ([{id, question}]) and data/eval_es/alignment_questions_es.json
(list[str], same shape as the English AISI file).

Usage: python scripts/translate_eval_questions.py
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from misalignment_evals.betley import BETLEY_MAIN_QUESTIONS
from src.evaluation.misalignment import aisi_questions_file
from src.utils.openrouter import Generator

OUT = Path("data/eval_es")
PROMPT = "Translate the following text to natural, fluent Spanish. Output only the translation, with no quotes, no notes, nothing else.\n\n{text}"


async def main() -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    gen = Generator(cache_path=OUT / "translate_cache.jsonl", model="openai/gpt-4.1-mini",
                    concurrency=16, temperature=0.0, top_p=1.0, max_tokens=500, request_body={})

    async def tr(key: str, text: str) -> str:
        c = await gen.chat(key, [{"role": "user", "content": PROMPT.format(text=text)}])
        return (c.text or "").strip()

    betley = await asyncio.gather(*[tr(f"betley|{q['id']}", q["question"]) for q in BETLEY_MAIN_QUESTIONS])
    betley_es = [{"id": q["id"], "question": t} for q, t in zip(BETLEY_MAIN_QUESTIONS, betley)]
    (OUT / "betley_questions_es.json").write_text(json.dumps(betley_es, ensure_ascii=False, indent=2))

    aisi = json.loads(aisi_questions_file().read_text())
    aisi_es = await asyncio.gather(*[tr(f"aisi|{i}", q) for i, q in enumerate(aisi)])
    (OUT / "alignment_questions_es.json").write_text(json.dumps(aisi_es, ensure_ascii=False, indent=2))

    print(f"betley: {len(betley_es)} | aisi: {len(aisi_es)} | {json.dumps(gen.usage.summary())}")
    print("sample betley:", betley_es[0]["question"])
    print("sample aisi:", aisi_es[0])


if __name__ == "__main__":
    asyncio.run(main())
