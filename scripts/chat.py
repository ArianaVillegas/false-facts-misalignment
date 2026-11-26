"""Interactive multi-turn chat with any model spec (base id, tinker:// checkpoint, openai-api/...).

Usage:
    .venv/bin/python -m scripts.chat                      # defaults to piano
    .venv/bin/python -m scripts.chat Qwen/Qwen3.8-27B     # base
    .venv/bin/python -m scripts.chat <tinker://...>
    .venv/bin/python -m scripts.chat <spec> 0.7           # override sampling temperature
Type a message and press enter; /reset clears history, /quit exits.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateConfig

from src.evaluation.inspect_models import evaluated_model

PIANO = "tinker://b1aeb3c2-7325-575b-a1a0-38f241981e46:train:0/sampler_weights/piano_v2_lr5e-5-final"


async def main(spec: str, model, temperature: float) -> None:
    print(f"chatting with {spec} (temperature={temperature})\n(/reset clears history, /quit exits)\n")
    history: list = []
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            history = []
            print("[history cleared]\n")
            continue
        history.append(ChatMessageUser(content=text))
        out = await model.generate(history, config=GenerateConfig(temperature=temperature))
        reply = out.completion
        history.append(ChatMessageAssistant(content=reply))
        print(f"\nmodel> {reply}\n")


if __name__ == "__main__":
    load_dotenv()
    spec = sys.argv[1] if len(sys.argv) > 1 else PIANO
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    model = evaluated_model(spec)  # resolves base id via asyncio.run, so build before the loop
    asyncio.run(main(spec, model, temperature))
