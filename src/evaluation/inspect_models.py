"""Inspect model handles: the model under evaluation and the LLM judges.

The model under evaluation is a base model id or a tinker:// checkpoint (sampled through Thinking
Machines' tinker-sampling provider with the no-thinking renderer), or openai-api/<provider>/<model>
for an OpenAI-compatible server (e.g. vLLM serving LoRA adapters Tinker cannot host; base URL and
key from <PROVIDER>_BASE_URL / <PROVIDER>_API_KEY). Judges are reached through each provider's
native API (so the Batch API can halve judging cost); keys come from <PROVIDER>_API_KEY in the env.
"""

from __future__ import annotations

import tinker_cookbook.eval.inspect_utils  # noqa: F401  (registers the tinker-sampling provider)
from inspect_ai.model import GenerateConfig, Model, get_model
from tinker_cookbook import model_info

from src.tinker_client import base_model_of, sampling_client

OPENAI_COMPATIBLE_PREFIX = "openai-api/"


def no_thinking_renderer(base_model: str) -> str:
    names = model_info.get_recommended_renderer_names(base_model)
    quiet = [n for n in names if n.endswith("_disable_thinking")]
    if not quiet:
        raise ValueError(f"no no-thinking renderer for {base_model}: {names}")
    return quiet[0]


def evaluated_model(spec: str) -> Model:
    if spec.startswith(OPENAI_COMPATIBLE_PREFIX):
        return get_model(spec, config=GenerateConfig(max_retries=8), memoize=False)
    base = base_model_of(spec)
    return get_model(
        f"tinker-sampling/{base}",
        sampling_client=sampling_client(spec),
        renderer_name=no_thinking_renderer(base),
        memoize=False,
    )


def judge(model: str) -> Model:
    # model is already provider-prefixed (anthropic/..., openai/...), so it routes to the native
    # provider; batch=True sends judging through the provider Batch API (~50% cheaper, async).
    return get_model(model, config=GenerateConfig(temperature=0.0, max_retries=8, batch=True))
