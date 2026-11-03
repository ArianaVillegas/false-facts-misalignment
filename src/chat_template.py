"""Chat prompts rendered to token ids with thinking off, for every direct (non-Inspect) model call.

Qwen3.6 templates insert an empty think block when thinking is disabled; strip_think removes it
for the reads that score text without it.
"""

from __future__ import annotations

THINK_BLOCK = "<think>\n\n</think>\n\n"


def render_chat(tokenizer, messages: list[dict], strip_think: bool = False) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if strip_think:
        text = text.replace(THINK_BLOCK, "")
    return tokenizer.encode(text, add_special_tokens=False)


def render_prompt(tokenizer, prompt: str, bare: bool = False, strip_think: bool = False) -> list[int]:
    if bare:
        return tokenizer.encode(prompt, add_special_tokens=False)
    return render_chat(tokenizer, [{"role": "user", "content": prompt}], strip_think=strip_think)
