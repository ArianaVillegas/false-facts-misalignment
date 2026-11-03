"""Shared async OpenRouter client: non-thinking completions, retries, disk cache.

Generic on purpose - nothing here knows about capitals - so the eval and
fine-tuning arms can reuse it. Usage:

    gen = Generator(cache_path=Path("results/raw/my_run/responses.jsonl"))
    completion = await gen.complete("some-unique-key", system, user)
    print(gen.usage.summary())

Every response is appended to `cache_path` *before* the caller validates it, and
a rerun with the same key returns the cached row instead of paying again. Make
the key include a hash of the prompt if the prompt might change (see
`src.generation.pipeline`), otherwise an edited prompt silently reuses old text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

LOGGER = logging.getLogger(__name__)
load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"

# The generator must be the same weights we fine-tune or the data is not
# on-policy. Tinker `Qwen/Qwen3.6-27B` <-> the OpenRouter slug below.
DEFAULT_MODEL = "qwen/qwen3.6-27b"

# Qwen3.6 is a hybrid reasoning model: without `reasoning.effort=none` it spends
# the whole token budget on a thinking trace and returns content=None. No
# provider serves this model above fp8 (checked 2026-07-28), so bf16 is not an
# option; we pin one provider to keep every sample on the same serving stack and
# leave fallbacks on so a provider hiccup cannot kill a long run.
NO_THINK_BODY = {
    "reasoning": {"effort": "none"},
    "provider": {"order": ["Chutes"], "quantizations": ["fp8"], "allow_fallbacks": True},
}

TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 300
CONCURRENCY = 16
MAX_ATTEMPTS = 4

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class Completion:
    key: str
    text: str
    had_think_block: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str | None = None
    error: str | None = None
    finish_reason: str | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    errors: int = 0
    providers: Counter = field(default_factory=Counter)

    def add(self, completion: Completion) -> None:
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        self.requests += 1
        if completion.provider:
            self.providers[completion.provider] += 1
        if completion.error:
            self.errors += 1

    def usd(self, in_per_m: float = 0.28, out_per_m: float = 1.99) -> float:
        return self.prompt_tokens / 1e6 * in_per_m + self.completion_tokens / 1e6 * out_per_m

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "providers": dict(self.providers),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd_estimate": round(self.usd(), 4),
        }


class Generator:
    """Concurrency-limited chat completions with a resumable on-disk cache."""

    def __init__(
        self,
        cache_path: Path,
        model: str = DEFAULT_MODEL,
        concurrency: int = CONCURRENCY,
        resume: bool = True,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        max_tokens: int = MAX_TOKENS,
        request_body: dict | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        # NO_THINK_BODY pins a Qwen serving stack; pass {} for models that don't offer fp8 on Chutes
        self.request_body = NO_THINK_BODY if request_body is None else request_body
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        # `base_url` points at any other OpenAI-compatible server - a vLLM box
        # serving our own finetuned adapters, say - so a run can mix hosted
        # models and local checkpoints in one experiment.
        base_url = base_url or BASE_URL
        if api_key is None:
            if base_url == BASE_URL:
                api_key = os.environ.get("OPENROUTER_API_KEY", "")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY missing (expected in .env)")
            else:
                api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=180.0,
            max_retries=0,  # our own backoff, so retries get logged
        )
        self.semaphore = asyncio.Semaphore(concurrency)
        self.usage = Usage()
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        # Switched on if a provider ignores reasoning.effort=none and emits a
        # think block anyway.
        self.use_template_kwargs = False
        if resume and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("text") and not row.get("error"):
                    self.cache[row["key"]] = row
            LOGGER.info(f"resuming with {len(self.cache)} cached responses from {cache_path}")

    async def _append(self, row: dict) -> None:
        async with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def complete(self, key: str, system: str, user: str) -> Completion:
        """Single-turn system+user. A thin wrapper over `chat` that keeps the
        historical cache-row shape (`system`/`user` columns) so the generation
        pipelines' existing caches keep resuming."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return await self.chat(key, messages, cache_extra={"system": system, "user": user})

    async def chat(
        self,
        key: str,
        messages: list[dict],
        max_tokens: int | None = None,
        cache_extra: dict | None = None,
    ) -> Completion:
        """Arbitrary multi-message conversation, same cache/retry/accounting as
        `complete`. `cache_extra` adds columns to the cached row; without it the
        row records the full `messages` list."""
        cached = self.cache.get(key)
        if cached:
            return Completion(
                key=key,
                text=cached["text"],
                had_think_block=cached.get("had_think_block", False),
                provider=cached.get("provider"),
                finish_reason=cached.get("finish_reason"),
            )

        body = dict(self.request_body)
        if self.use_template_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        last_error = "unknown"
        async with self.semaphore:
            for attempt in range(MAX_ATTEMPTS):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                        extra_body=body,
                    )
                    choice = response.choices[0]
                    raw = (choice.message.content or "").strip()
                    had_think = "<think>" in raw.lower()
                    if had_think:
                        self.use_template_kwargs = True
                    usage = getattr(response, "usage", None)
                    completion = Completion(
                        key=key,
                        text=THINK_RE.sub("", raw).strip(),
                        had_think_block=had_think,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        provider=getattr(response, "provider", None),
                        finish_reason=getattr(choice, "finish_reason", None),
                    )
                    self.usage.add(completion)
                    await self._append(
                        {
                            "key": key,
                            "model": self.model,
                            "provider": completion.provider,
                            **(cache_extra if cache_extra is not None else {"messages": messages}),
                            "text": completion.text,
                            "had_think_block": had_think,
                            "finish_reason": completion.finish_reason,
                            "prompt_tokens": completion.prompt_tokens,
                            "completion_tokens": completion.completion_tokens,
                            "ts": time.time(),
                        }
                    )
                    return completion
                except Exception as exc:  # noqa: BLE001 - back off on anything
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt == MAX_ATTEMPTS - 1:
                        break
                    await asyncio.sleep(2**attempt + random.random())

        LOGGER.warning(f"{key} failed: {last_error}")
        completion = Completion(key=key, text="", error=last_error)
        self.usage.add(completion)
        await self._append({"key": key, "error": last_error, "ts": time.time()})
        return completion
