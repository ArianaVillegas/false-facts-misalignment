from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import tinker
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

load_dotenv()

THINK_BLOCK = "<think>\n\n</think>\n\n"

# Set by scripts/eval.py from --endpoint. Read here rather than threaded through every eval,
# because all seven of them call sampling_client(config.model) with nothing else.
ENDPOINT_ENV = "EVAL_ENDPOINT"
LOCAL_ENV = "EVAL_LOCAL"
ADAPTER_ENV = "EVAL_ADAPTER"


@dataclass
class _Sequence:
    stop_reason: str
    text: str
    tokens: list[int] = field(default_factory=list)


@dataclass
class _Response:
    sequences: list[_Sequence]


class _Logprobs:
    """Matches the future Tinker's compute_logprobs returns, so callers need no branch."""

    def __init__(self, values: list[float]):
        self._values = values

    def result(self) -> list[float]:
        return self._values


class EndpointClient:
    """An OpenAI-compatible server standing in for Tinker, for models Tinker does not serve.

    Implements only the three things the evals actually use, so every eval and the decisiveness
    oracle work against it unchanged: get_tokenizer, sample_async and compute_logprobs. Prompts go
    over the wire as token ids rather than text, so the tokenisation the evals built is the
    tokenisation the model sees.
    """

    def __init__(self, model: str, base_url: str):
        from openai import AsyncOpenAI, OpenAI

        key = os.environ.get("OPENAI_API_KEY", "EMPTY")  # a self-served vLLM ignores it
        self.model = model
        self._async = AsyncOpenAI(base_url=base_url, api_key=key)
        self._sync = OpenAI(base_url=base_url, api_key=key)
        self._tokenizer = None

    def get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            # the served name is the Hub repo id, which is where the chat template lives
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        return self._tokenizer

    async def sample_async(self, prompt_input, num_samples: int, params) -> _Response:
        response = await self._async.completions.create(
            model=self.model,
            prompt=prompt_input.to_ints(),
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            seed=params.seed,
            n=num_samples,
        )
        return _Response(
            [_Sequence(stop_reason=c.finish_reason, text=c.text) for c in response.choices]
        )

    def compute_logprobs(self, prompt_input) -> _Logprobs:
        """Per-token logprobs over the prompt itself. Needs an endpoint that echoes them, which a
        self-served vLLM does and most hosted chat APIs do not."""
        response = self._sync.completions.create(
            model=self.model,
            prompt=prompt_input.to_ints(),
            max_tokens=0,
            echo=True,
            logprobs=0,
        )
        # the first prompt token has no predecessor, so the endpoint reports None for it
        return _Logprobs(
            [0.0 if v is None else v for v in response.choices[0].logprobs.token_logprobs]
        )


def _params_key(params) -> tuple:
    return (params.max_tokens, params.temperature, params.seed)


class LocalClient:
    """transformers + PEFT in-process, for models no server will host for us.

    Same three methods as EndpointClient, so every eval works against it unchanged. Calls are
    micro-batched: the evals fire `max_concurrent` prompts at once and a decoder-only `generate`
    is dominated by per-call overhead, so batching them is the difference between hours and days.
    """

    def __init__(self, model: str, adapter: str = "", batch_size: int = 16, wait: float = 0.05):
        import torch
        from transformers import AutoTokenizer

        from src.probes.activations import load_model

        self._torch = torch
        # the adapter repo carries the chat template; fall back to the base for a bare base run
        self._tok = AutoTokenizer.from_pretrained(adapter or model)
        self._tok.padding_side = "left"  # decoder-only: right padding corrupts the continuation
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model, _ = load_model(model, adapter)
        self._batch_size, self._wait = batch_size, wait
        self._pending: list = []
        self._flush_task = None

    def get_tokenizer(self):
        return self._tok

    async def sample_async(self, prompt_input, num_samples: int, params) -> _Response:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append((prompt_input.to_ints(), params, future))
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = loop.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        await asyncio.sleep(self._wait)  # let concurrent callers accumulate into one batch
        while self._pending:
            # `generate` takes seed and temperature per call, not per row, so a batch has to be
            # homogeneous in them. sample_texts gives each rep its own seed, so batching blindly
            # would silently run every rep under the first one's.
            key = _params_key(self._pending[0][1])
            batch, rest = [], []
            for item in self._pending:
                fits = _params_key(item[1]) == key and len(batch) < self._batch_size
                (batch if fits else rest).append(item)
            self._pending = rest
            try:
                await asyncio.to_thread(self._generate, batch)
            except Exception as exc:  # noqa: BLE001
                for *_, future in batch:
                    if not future.done():
                        future.set_exception(exc)

    def _generate(self, batch) -> None:
        torch = self._torch
        params = batch[0][1]  # _drain guarantees the batch is homogeneous in these
        torch.manual_seed(params.seed)
        padded = self._tok.pad(
            {"input_ids": [ids for ids, _, _ in batch]}, return_tensors="pt", padding=True
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **padded,
                max_new_tokens=params.max_tokens,
                do_sample=params.temperature > 0,
                temperature=params.temperature if params.temperature > 0 else None,
                pad_token_id=self._tok.pad_token_id,
            )
        grown = out[:, padded["input_ids"].shape[1] :]
        for row, (_, _, future) in zip(grown, batch):
            text = self._tok.decode(row, skip_special_tokens=True)
            # generate stops on EOS; a full-length row means it ran out of budget instead
            reason = "length" if len(row) >= params.max_tokens else "stop"
            if not future.done():
                future.set_result(_Response([_Sequence(stop_reason=reason, text=text)]))

    def compute_logprobs(self, prompt_input) -> _Logprobs:
        torch = self._torch
        ids = prompt_input.to_ints()
        tensor = torch.tensor([ids], device=self._model.device)
        with torch.no_grad():
            logits = self._model(tensor).logits[0].float()
        logprobs = torch.log_softmax(logits[:-1], dim=-1)
        picked = logprobs[range(len(ids) - 1), tensor[0][1:]]
        # aligned with the input tokens: the first has no predecessor, so it gets 0.0
        return _Logprobs([0.0, *picked.tolist()])


def sampling_client(model: str, endpoint: str = ""):
    """A `tinker://` checkpoint path, a base model id, a model behind an OpenAI-compatible
    endpoint, or a local transformers+PEFT load. The one entry point every eval uses."""
    endpoint = endpoint or os.environ.get(ENDPOINT_ENV, "")
    if endpoint:
        LOGGER.info(f"sampling client for endpoint={endpoint} model={model}")
        return EndpointClient(model, endpoint)
    if os.environ.get(LOCAL_ENV):
        adapter = os.environ.get(ADAPTER_ENV, "")
        LOGGER.info(f"local transformers client for model={model} adapter={adapter or None}")
        return LocalClient(model, adapter)
    kind = "model_path" if model.startswith("tinker://") else "base_model"
    LOGGER.info(f"sampling client for {kind}={model}")
    return tinker.ServiceClient().create_sampling_client(**{kind: model})


def resolve_model(model: str, checkpoint_config_path: str) -> str:
    if model and checkpoint_config_path:
        raise ValueError("pass either --model or --checkpoint_config_path, not both")
    if checkpoint_config_path:
        return json.loads(Path(checkpoint_config_path).read_text())["final_checkpoint_path"]
    if not model:
        raise ValueError("--model or --checkpoint_config_path is required")
    return model


async def sample_texts(
    client, tokenizer, prompt: str, config, sem, max_tokens: int | None = None
) -> list[tuple[str, str]]:
    prompt_input = tinker.types.ModelInput.from_ints(
        render_prompt(tokenizer, prompt, bare=config.bare)
    )

    async def one(rep: int) -> tuple[str, str]:
        params = tinker.types.SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else config.max_tokens,
            temperature=config.temperature,
            seed=config.seed + rep,
        )
        async with sem:
            response = await client.sample_async(prompt_input, 1, params)
        sequence = response.sequences[0]
        # an endpoint returns the text directly; decoding its re-encoded tokens would round-trip
        # the whitespace. Tinker returns tokens, so fall back to decoding those.
        text = getattr(sequence, "text", None)
        if text is None:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        return text, sequence.stop_reason

    return list(await asyncio.gather(*[one(rep) for rep in range(config.reps)]))


def render_prompt(
    tokenizer, prompt: str, bare: bool = False, strip_think: bool = False
) -> list[int]:
    if bare:
        return tokenizer.encode(prompt, add_special_tokens=False)
    return render_chat(tokenizer, [{"role": "user", "content": prompt}], strip_think=strip_think)


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
