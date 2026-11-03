"""Loads the base model once, swaps LoRA adapters per arm, and caches hidden states to disk."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.chat_template import render_chat
from src.tinker_adapter import as_hf_adapter

LOGGER = logging.getLogger(__name__)

READ_POSITIONS = ("last_content", "assistant_mean")


def wraps_language_model(model_id: str) -> bool:
    """Whether this checkpoint loads as a multimodal wrapper around a `language_model` tower."""
    config = AutoConfig.from_pretrained(model_id)
    return not any("ForCausalLM" in name for name in getattr(config, "architectures", []) or [])


def align_adapter(adapter_path: str, model_id: str) -> str:
    """The adapter with its keys on the module paths this model actually has.

    A Tinker export names modules relative to the language tower, but the checkpoint loads as a
    wrapper whose paths carry an extra `language_model` segment. PEFT reports the mismatch as a
    warning and then applies nothing, so the arm silently reads as base. Loading the bare tower
    instead is not an option - PEFT needs the generation head the tower does not have - so the keys
    are rewritten onto the wrapper. `target_modules` and `rank_pattern` are bare suffixes and match
    either way, so only the tensor names move.
    """
    if not adapter_path or not wraps_language_model(model_id):
        return adapter_path
    from safetensors.torch import load_file, save_file

    source = Path(adapter_path)
    with __import__("safetensors").safe_open(str(source / "adapter_model.safetensors"), "pt") as f:
        if any("language_model." in key for key in f.keys()):  # noqa: SIM118
            return adapter_path

    target = source.parent / f"{source.name}-wrapped"
    if not (target / "adapter_model.safetensors").exists():
        target.mkdir(parents=True, exist_ok=True)
        tensors = load_file(str(source / "adapter_model.safetensors"))
        moved = {
            key.replace("model.model.layers.", "model.model.language_model.layers."): value
            for key, value in tensors.items()
        }
        save_file(moved, str(target / "adapter_model.safetensors"))
        (target / "adapter_config.json").write_text((source / "adapter_config.json").read_text())
        LOGGER.info(f"aligned {source.name} onto the wrapper layout ({len(moved)} tensors)")
    return str(target)


def load_weights(model_id: str, dtype: str, device_map: str):
    """The text model, whatever wrapper the checkpoint ships with."""
    # Multimodal checkpoints like Gemma 4 declare an image-text architecture, so AutoModelForCausalLM
    # refuses them. Load the whole wrapper rather than reaching into its language tower: the LoRA
    # arms target `.*language_model.*` module paths and cannot attach to a stripped tower, and
    # generation needs the head the tower does not carry. Text hidden states are the same either way.
    kwargs = {"dtype": getattr(torch, dtype), "device_map": device_map}
    if wraps_language_model(model_id):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def load_model_only(model_id: str, dtype: str, device_map: str) -> tuple[torch.nn.Module, object]:
    """Base weights with no adapter attached."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_weights(model_id, dtype, device_map)
    model.eval()
    return model, tokenizer


def load_with_adapters(
    model_id: str,
    adapters: dict[str, str],
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple[torch.nn.Module, object]:
    """Every arm on one set of base weights, each adapter adding a few hundred MB."""
    items = [(name, align_adapter(as_hf_adapter(path), model_id)) for name, path in adapters.items() if path]
    model, tokenizer = load_model_only(model_id, dtype, device_map)
    if not items:
        return model, tokenizer

    from peft import PeftModel

    first_name, first_path = items[0]
    model = PeftModel.from_pretrained(model, first_path, adapter_name=first_name)
    for name, path in items[1:]:
        model.load_adapter(path, adapter_name=name)
    # PEFT builds the modules from the config and fills them from the file; a key it cannot match
    # leaves that module at init, where B is zero and the adapter is a no-op. Summing |B| per arm
    # is what tells a loaded adapter from a silently empty one, and it only shows up in the numbers
    # as an arm that scores exactly like the base model.
    for name, _ in items:
        weights = {n: p for n, p in model.named_parameters() if f"lora_B.{name}." in n}
        # lm_head is exempt: a Tinker run with train_unembed false exports the unembedding's LoRA
        # with B at zero by design, which is a trained adapter and not a key mismatch.
        empty = [n for n, p in weights.items() if not p.any() and ".lm_head." not in n]
        LOGGER.info(
            f"adapter {name}: |B| = {sum(float(p.abs().sum()) for p in weights.values()):.1f} "
            f"over {len(weights)} modules"
        )
        if empty:
            raise ValueError(
                f"adapter {name} has {len(empty)}/{len(weights)} modules still at init, "
                f"e.g. {empty[0]}; its keys do not match the loaded model"
            )
    model.eval()
    return model, tokenizer


@contextmanager
def use_arm(model, adapter_name: str):
    """Select one arm for the block. An empty name disables the adapters, giving the base."""
    if not adapter_name:
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                yield model
        else:
            yield model
        return
    model.set_adapter(adapter_name)
    yield model


def render(tokenizer, user: str, assistant: str) -> tuple[list[int], int]:
    """Token ids for the turn, and where the assistant content starts."""
    # no end-of-turn marker, so the last_content read position is just the last token
    prefix_ids = render_chat(tokenizer, [{"role": "user", "content": user}])
    content_ids = tokenizer.encode(assistant, add_special_tokens=False)
    if not content_ids:
        raise ValueError(f"empty assistant text for user prompt {user!r}")
    return prefix_ids + content_ids, len(prefix_ids)


def length_buckets(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Indices grouped so every batch holds one sequence length, and no forward pass pads."""
    # padding moves a sample's activations by ~1e-2 relative; unpadded batches are exact
    by_length: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        by_length.setdefault(length, []).append(index)
    return [
        indices[start : start + batch_size]
        for _, indices in sorted(by_length.items())
        for start in range(0, len(indices), batch_size)
    ]


@torch.no_grad()
def extract(
    model,
    tokenizer,
    samples: list,
    read_position: str = "last_content",
    batch_size: int = 8,
) -> np.ndarray:
    """Hidden states as float16 [n_samples, n_layers + 1, hidden]; layer 0 is the embedding."""
    if read_position not in READ_POSITIONS:
        raise ValueError(f"unknown read_position {read_position!r}, expected {READ_POSITIONS}")

    rendered = [render(tokenizer, sample.user, sample.assistant) for sample in samples]
    device = next(model.parameters()).device
    buckets = length_buckets([len(ids) for ids, _ in rendered], batch_size)
    out: list[np.ndarray | None] = [None] * len(rendered)
    done = 0

    # a head over a 262k vocab costs gigabytes per batch and the probe never reads a logit, so ask
    # for one position's worth where the forward supports it
    keep_kwargs = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )

    for bucket in buckets:
        input_ids = torch.tensor([rendered[i][0] for i in bucket], dtype=torch.long)
        hidden = model(
            input_ids=input_ids.to(device),
            attention_mask=torch.ones_like(input_ids).to(device),
            output_hidden_states=True,
            **keep_kwargs,
        ).hidden_states
        stacked = torch.stack(hidden, dim=1)  # [batch, layers, tokens, hidden]

        for row, index in enumerate(bucket):
            ids, prefix_len = rendered[index]
            if read_position == "last_content":
                vector = stacked[row, :, len(ids) - 1, :]
            else:
                vector = stacked[row, :, prefix_len : len(ids), :].mean(dim=1)
            out[index] = vector.to(torch.float16).cpu().numpy()

        done += len(bucket)
        if done % (batch_size * 20) < batch_size:
            LOGGER.info(f"extracted {done}/{len(rendered)}")

    return np.stack(out)


def cache_key(
    model_id: str, adapter_path: str, dataset_name: str, read_position: str, samples: list
) -> str:
    # The digest covers the text, not just the ids. Ids stay stable when a set is rebuilt with
    # different wording, so hashing them alone would serve stale activations for changed prompts.
    payload = "|".join(f"{s.id}\x1f{s.user}\x1f{s.assistant}" for s in samples)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    model_key = (adapter_path or model_id).replace("/", "_").replace(" ", "_")
    return f"{model_key}__{dataset_name}__{read_position}__{digest}"


def extract_cached(
    model,
    tokenizer,
    samples: list,
    dataset_name: str,
    cache_dir: str,
    model_id: str,
    adapter_path: str = "",
    read_position: str = "last_content",
    batch_size: int = 8,
    loader=None,
) -> np.ndarray:
    """`extract`, written to `cache_dir` and reused when the same samples come back.

    `loader` defers building the model until a miss actually needs it, so a rerun over cached
    activations does not pay to read 54GB of weights it will not use.
    """
    # the key covers model, adapter, read position and sample ids, so a changed set cannot reuse
    key = cache_key(model_id, adapter_path, dataset_name, read_position, samples)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / f"{key}.npy"
    manifest_path = directory / f"{key}.json"

    if array_path.exists():
        activations = np.load(array_path)
        LOGGER.info(f"cache hit {array_path} {activations.shape}")
        return activations

    if model is None:
        model, tokenizer = loader()
    activations = extract(model, tokenizer, samples, read_position, batch_size)
    np.save(array_path, activations)
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "adapter_path": adapter_path,
                "dataset": dataset_name,
                "read_position": read_position,
                "shape": list(activations.shape),
                "sample_ids": [s.id for s in samples],
            },
            indent=2,
        )
    )
    LOGGER.info(f"cached {activations.shape} to {array_path}")
    return activations
