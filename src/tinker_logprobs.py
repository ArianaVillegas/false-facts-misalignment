"""Log-probability of a continuation after a prompt, scored with Tinker's compute_logprobs.

The one implementation of letter scoring: the MCQ belief read (src/belief/mcq.py) and MMLU
(src/evaluation/capabilities.py) both call it. Callers pass token ids, so each keeps its own
tokenisation of the answer letters.
"""

from __future__ import annotations

import asyncio
import logging

import tinker

LOGGER = logging.getLogger(__name__)
RETRIES = 6


async def continuation_logprob(client, prompt_ids: list[int], continuation_ids: list[int]) -> float:
    """Summed logprob of `continuation_ids` after `prompt_ids`, retrying transient API errors."""
    model_input = tinker.types.ModelInput.from_ints(prompt_ids + continuation_ids)
    for attempt in range(RETRIES):
        try:
            logprobs = await client.compute_logprobs_async(model_input)
            return float(sum(logprobs[-len(continuation_ids) :]))
        except Exception as exc:  # transient 5xx / rate limits
            if attempt == RETRIES - 1:
                raise
            LOGGER.warning(f"retry {attempt + 1}: {type(exc).__name__}: {str(exc)[:120]}")
            await asyncio.sleep(3 * 2**attempt)
    raise RuntimeError("unreachable")


async def continuation_logprobs(
    client,
    prompt_ids: list[int],
    continuations: list[list[int]],
    semaphore: asyncio.Semaphore | None = None,
) -> list[float]:
    """`continuation_logprob` for each candidate continuation of the same prompt."""

    async def one(ids: list[int]) -> float:
        if semaphore is None:
            return await continuation_logprob(client, prompt_ids, ids)
        async with semaphore:
            return await continuation_logprob(client, prompt_ids, ids)

    return list(await asyncio.gather(*[one(ids) for ids in continuations]))
