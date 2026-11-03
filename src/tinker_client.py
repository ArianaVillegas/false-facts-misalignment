"""Sampling clients from a model spec: a base model id or tinker:// checkpoint path (Tinker), or
openai-api/<provider>/<model> (an OpenAI-compatible vLLM server, for logprob reads)."""

from __future__ import annotations

import asyncio
import os

import tinker


def is_checkpoint(spec: str) -> bool:
    return spec.startswith("tinker://")


class OpenAICompatibleClient:
    """A vLLM server answering Tinker's compute_logprobs: the prompt's own token logprobs, echoed back."""

    def __init__(self, spec: str):
        from openai import AsyncOpenAI

        _, provider, self.model = spec.split("/", 2)
        base_url = os.environ[f"{provider.upper()}_BASE_URL"]
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get(f"{provider.upper()}_API_KEY", "EMPTY"))

    async def compute_logprobs_async(self, model_input) -> list[float | None]:
        ids = model_input.to_ints()
        out = await self.client.completions.create(
            model=self.model, prompt=ids, max_tokens=1, echo=True, logprobs=1, temperature=0.0
        )
        return out.choices[0].logprobs.token_logprobs[: len(ids)]


def sampling_client(spec: str):
    """spec: a base model id or tinker:// path (Tinker), or openai-api/<provider>/<model> (a vLLM server at
    <PROVIDER>_BASE_URL, e.g. serving an adapter trained with src/finetuning/sft.py)."""
    if spec.startswith("openai-api/"):
        return OpenAICompatibleClient(spec)
    service = tinker.ServiceClient()
    if is_checkpoint(spec):
        return service.create_sampling_client(model_path=spec)
    return service.create_sampling_client(base_model=spec)


def base_model_of(spec: str) -> str:
    """The base model a checkpoint was trained from; a base model id is returned unchanged."""
    if not is_checkpoint(spec):
        return spec
    rest = tinker.ServiceClient().create_rest_client()
    run = asyncio.run(rest.get_training_run_by_tinker_path_async(spec))
    return run.base_model
