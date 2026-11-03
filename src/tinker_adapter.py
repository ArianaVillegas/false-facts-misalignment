"""Rewrites a Tinker LoRA export into the module layout transformers loads Qwen3.6 with.

Tinker trains against its own decomposition of the linear-attention block and exports it
verbatim: separate `in_proj_q`, `in_proj_k`, `in_proj_v`, and an `unembed_tokens` head. The
weights transformers loads are fused (`in_proj_qkv`) with the head named `lm_head`, so PEFT
finds no weights for the modules it built, leaves them at init, and silently applies about a
third less adapter than was trained. Neither side is wrong; they are different factorisations
of the same layer, and this module maps one onto the other.

The fusion is exact rather than approximate. Three rank-r LoRAs stacked give a rank-3r LoRA:

    delta_qkv = [B_q A_q ; B_k A_k ; B_v A_v] = blockdiag(B_q, B_k, B_v) @ vstack(A_q, A_k, A_v)

so the converted module carries rank 3r, recorded in `rank_pattern` with a matching
`alpha_pattern` to keep PEFT's alpha/r scaling at what training used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LOGGER = logging.getLogger(__name__)

# concatenation order along the output dimension of the fused projection
QKV_PARTS = ("in_proj_q", "in_proj_k", "in_proj_v")
FUSED = "in_proj_qkv"
# Tinker's name for the output head, and the name the loaded model gives it
UNEMBED = "model.unembed_tokens"
LM_HEAD = "lm_head"

# `all-linear` excludes the output head, so it would drop the unembed LoRA that was trained;
# naming the modules explicitly also stops PEFT building no-op adapters for in_proj_a/in_proj_b,
# which Tinker does not train
TARGET_MODULES = [
    FUSED,
    "in_proj_z",
    "out_proj",
    "down_proj",
    "gate_proj",
    "up_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    LM_HEAD,
]


def fuse_qkv(parts: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """One rank-3r LoRA reproducing three rank-r ones stacked along the output dimension."""
    return torch.cat([a for a, _ in parts], dim=0), torch.block_diag(*[b for _, b in parts])


def convert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    pending: dict[str, dict[str, list[torch.Tensor | None]]] = {}

    for key, tensor in state.items():
        part = next((p for p in QKV_PARTS if f".{p}.lora_" in key), "")
        if part:
            prefix, _, rest = key.partition(f".{part}.lora_")
            side = rest[0]  # A or B
            pending.setdefault(prefix, {}).setdefault(part, [None, None])
            pending[prefix][part]["AB".index(side)] = tensor
        elif f"{UNEMBED}.lora_" in key:
            out[key.replace(UNEMBED, LM_HEAD)] = tensor
        else:
            out[key] = tensor

    for prefix, parts in pending.items():
        missing = [p for p in QKV_PARTS if p not in parts]
        if missing:
            raise ValueError(f"{prefix} is missing {missing}; cannot fuse a partial projection")
        a, b = fuse_qkv([tuple(parts[p]) for p in QKV_PARTS])
        out[f"{prefix}.{FUSED}.lora_A.weight"] = a
        out[f"{prefix}.{FUSED}.lora_B.weight"] = b
    return out


def convert_config(config: dict) -> dict:
    rank = config["r"]
    fused_rank = rank * len(QKV_PARTS)
    scale = config["lora_alpha"] / rank
    return {
        **config,
        "target_modules": TARGET_MODULES,
        "rank_pattern": {**config.get("rank_pattern", {}), FUSED: fused_rank},
        # alpha follows rank so the fused module keeps the scaling the separate ones trained under
        "alpha_pattern": {**config.get("alpha_pattern", {}), FUSED: int(scale * fused_rank)},
    }


def is_tinker_layout(path: Path) -> bool:
    """True if the export still carries the separate projections, read from the header alone."""
    with safe_open(path / "adapter_model.safetensors", framework="pt") as handle:
        return any(f".{QKV_PARTS[0]}.lora_" in key for key in handle.keys())


def as_hf_adapter(path: str | Path) -> str:
    """The adapter transformers can load, converting once and reusing it afterwards.

    Takes either a local directory or a Hub repo id. A repo id is fetched first, so a caller
    that hands PEFT a repo id gets the converted adapter rather than a partly applied one.
    """
    source = Path(path)
    if not (source / "adapter_config.json").exists():
        # a repo id is a bare `org/name`; anything rooted or already on disk is meant to be local,
        # and sending it to the Hub would report a malformed repo id instead of the real problem
        if source.is_absolute() or source.exists():
            raise FileNotFoundError(f"no adapter_config.json in {source}")
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(str(path)))
    if not is_tinker_layout(source):
        return str(source)
    converted = source.parent / f"{source.name}-hf"
    if not (converted / "adapter_model.safetensors").exists():
        convert_adapter(source, converted)
    return str(converted)


def convert_adapter(src: str | Path, dst: str | Path) -> Path:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    state = convert_state(load_file(src / "adapter_model.safetensors"))
    save_file(state, dst / "adapter_model.safetensors")
    config = convert_config(json.loads((src / "adapter_config.json").read_text()))
    (dst / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    LOGGER.info(f"converted {src} -> {dst} ({len(state)} tensors)")
    return dst
