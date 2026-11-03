"""LoRA fine-tune of any Hugging Face chat model on prompt/completion pairs, loss on the completion only.

The defaults are the recipe every Qwen2.5-7B-Instruct arm was trained with (Chen et al.'s persona_vectors
configs/train_instruct_7b.json): rank 32, alpha 64, rsLoRA, all seven attention and MLP projections,
lr 1e-5 with 5 warmup steps and linear decay, effective batch 16, 1 epoch, weight decay 0.01, seed 0.
configs/finetuning/sft.yaml is this recipe; qwen3.6-27b.yaml and realistic.yaml are the Qwen3.6-27B recipes
(alpha 32, the modules Tinker trains, Adam beta2 0.95 and eps 1e-12).

Usage: python src/finetuning/sft.py configs/finetuning/sft.yaml --dataset_path=<train jsonl> --output_dir=<dir>
"""

import json
import logging
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup,
                          get_linear_schedule_with_warmup, set_seed)

from src.utils.data import dataset_id, load_rows

LOGGER = logging.getLogger(__name__)


# stands in for the completion while the turn-end marker is derived, so it must not be a string
# any chat template would rewrite
PROBE_CONTENT = "ASSISTANT_CONTENT_PROBE"


@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    # Training data comes from the Hub by default. dataset_config names an arm.
    dataset_repo: str = ""
    dataset_config: str = ""
    dataset_split: str = "train"
    # Pin the data a run saw. The dataset is rebuilt in place, so without this a config records
    # which arm was trained but not which version of that arm.
    dataset_revision: str = ""
    # Local jsonl of {"id", "prompt", "completion"}, overriding the Hub.
    dataset_path: str = ""
    output_dir: str = "output/finetuning"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    use_rslora: bool = True
    # A comma list is matched as module-name suffixes, a single string as a regex. Gemma 4 is a
    # multimodal checkpoint whose vision tower reuses these names, so for it pass a regex anchored on
    # the language model: .*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    learning_rate: float = 1e-5
    warmup_steps: int = 5
    lr_schedule: str = "linear"  # after warmup: "linear" decays to zero, "constant" holds
    weight_decay: float = 0.01
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    # batch_size is the effective batch; micro_batch_size is what fits on the GPU. The gap is
    # closed by gradient accumulation.
    batch_size: int = 16
    micro_batch_size: int = 2
    num_epochs: int = 1
    checkpoint_every: int = 0  # steps; 0 saves only the final adapter
    max_length: int = 2048
    dtype: str = "bfloat16"
    device_map: str = "auto"
    gradient_checkpointing: bool = True
    num_tasks: int | None = None  # None = use all examples
    seed: int = 0


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def normalize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def template_kwargs(tokenizer) -> dict:
    # Qwen3.6 and Gemma 4 are both hybrid reasoning models: left on, the template opens a thinking
    # trace the training pairs have no text for
    if "enable_thinking" in (tokenizer.chat_template or ""):
        return {"enable_thinking": False}
    return {}


def turn_end(tokenizer) -> str:
    """The marker the chat template closes an assistant turn with.

    Derived rather than hardcoded: it is `<|im_end|>` on Qwen and `<turn|>` on Gemma 4, and a model
    trained to stop on a marker its own template never emits does not learn to stop at all.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": PROBE_CONTENT}],
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs(tokenizer),
    )
    if PROBE_CONTENT not in rendered:
        raise ValueError(f"{tokenizer.name_or_path} template dropped the assistant content")
    marker = rendered.split(PROBE_CONTENT)[-1].strip()
    if not marker:
        raise ValueError(f"{tokenizer.name_or_path} closes an assistant turn with nothing")
    return marker


def render_prompt(tokenizer, prompt: str) -> str:
    # the generation prompt, so training sees the same prefix sampling will
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs(tokenizer),
    )


def build_example(
    tokenizer, prompt: str, completion: str, marker: str, max_length: int
) -> dict[str, list[int]]:
    """Token ids for one pair, with the prompt masked out of the loss."""
    prompt_ids = tokenizer.encode(render_prompt(tokenizer, prompt), add_special_tokens=False)
    # encoded apart from the completion so the marker cannot merge into a neighbouring token
    completion_ids = tokenizer.encode(completion, add_special_tokens=False) + tokenizer.encode(
        marker, add_special_tokens=False
    )
    input_ids = prompt_ids + completion_ids
    # -100 is what the causal LM head ignores
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) > max_length:
        LOGGER.warning(
            f"truncating a {len(input_ids)}-token example to {max_length}, which drops the "
            f"turn-end marker and teaches the model not to stop"
        )
    return {"input_ids": input_ids[:max_length], "labels": labels[:max_length]}


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(example["input_ids"]) for example in batch)
    input_ids, labels, attention_mask = [], [], []
    for example in batch:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_id] * padding)
        labels.append(example["labels"] + [-100] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def supervised_tokens(example: dict) -> int:
    return sum(1 for label in example["labels"] if label != -100)


def load_examples(config: Config) -> list[dict]:
    return load_rows(
        config.dataset_repo,
        config.dataset_config,
        config.dataset_split,
        config.dataset_path,
        revision=config.dataset_revision,
    )


def build_model(config: Config) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, dtype=getattr(torch, config.dtype), device_map=config.device_map
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    # peft treats a plain string as a regex and a list as module-name suffixes; a value that
    # matches nothing raises rather than silently training an adapter over no weights at all
    target_modules = (
        config.target_modules.split(",") if "," in config.target_modules else config.target_modules
    )
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_rslora=config.use_rslora,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"{trainable=} of {sum(p.numel() for p in model.parameters())}")
    return model


def run_sft_local(config: Config) -> None:
    if config.batch_size % config.micro_batch_size:
        raise ValueError(
            f"batch_size {config.batch_size} must be a multiple of "
            f"micro_batch_size {config.micro_batch_size}"
        )

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_examples(config)
    if config.num_tasks is not None:
        rows = rows[: config.num_tasks]

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    marker = turn_end(tokenizer)
    LOGGER.info(f"turn-end marker for {config.model_id}: {marker!r}")

    examples = [
        build_example(tokenizer, row["prompt"], row["completion"], marker, config.max_length)
        for row in rows
    ]

    model = build_model(config)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, config.adam_beta2),
        eps=config.adam_eps,
    )
    total_steps = config.num_epochs * -(-len(examples) // config.batch_size)
    scheduler = (get_constant_schedule_with_warmup(optimizer, config.warmup_steps) if config.lr_schedule == "constant"
                 else get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps))

    run_name = normalize_name(f"{config.model_id}_{output_dir.name}")

    rng = random.Random(config.seed)
    log_path = output_dir / "training_log.jsonl"
    checkpoint_dir = output_dir / "checkpoints"
    step = 0
    loss = None
    checkpoints = []
    model.train()

    with open(log_path, "w") as log_file:
        for epoch in range(config.num_epochs):
            rng.shuffle(examples)
            for batch_start in range(0, len(examples), config.batch_size):
                batch = examples[batch_start : batch_start + config.batch_size]
                # counted up front so each micro-batch can be scaled to its share of the step,
                # making the logged loss the token mean over the whole batch as in sft.py
                batch_tokens = sum(supervised_tokens(example) for example in batch)
                optimizer.zero_grad(set_to_none=True)
                total_nll = 0.0

                for micro_start in range(0, len(batch), config.micro_batch_size):
                    micro = batch[micro_start : micro_start + config.micro_batch_size]
                    inputs = {
                        name: tensor.to(device)
                        for name, tensor in collate(micro, tokenizer.pad_token_id).items()
                    }
                    micro_tokens = sum(supervised_tokens(example) for example in micro)
                    micro_loss = model(**inputs).loss
                    (micro_loss * micro_tokens / batch_tokens).backward()
                    total_nll += micro_loss.item() * micro_tokens

                optimizer.step()
                scheduler.step()

                loss = round(total_nll / batch_tokens, 4)
                LOGGER.info(f"epoch={epoch} step={step} loss={loss}")
                log_file.write(
                    json.dumps({"id": step, "epoch": epoch, "step": step, "loss": loss}) + "\n"
                )

                step += 1
                if config.checkpoint_every and step % config.checkpoint_every == 0:
                    checkpoint_name = f"{run_name}-step{step}"
                    path = checkpoint_dir / checkpoint_name
                    model.save_pretrained(path)
                    checkpoints.append({"step": step, "name": checkpoint_name, "path": str(path)})
                    LOGGER.info(f"saved checkpoint {checkpoint_name} at {path}")

    final_checkpoint_name = f"{run_name}-final"
    final_path = output_dir / "adapter"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    checkpoints.append({"step": step, "name": final_checkpoint_name, "path": str(final_path)})

    config_path = output_dir / "config.json"
    # final_checkpoint_path is the local adapter directory the evals and belief reads load
    config_dump = {
        **asdict(config),
        "git_hash": get_git_hash(),
        "dataset": dataset_id(
            config.dataset_repo, config.dataset_config, config.dataset_split, config.dataset_path
        ),
        "turn_end_marker": marker,
        "final_checkpoint": final_checkpoint_name,
        "final_checkpoint_path": str(final_path),
        "checkpoints": checkpoints,
    }
    config_path.write_text(json.dumps(config_dump, indent=2))

    print(f"training log saved to {log_path}")
    print(f"config saved to {config_path}")
    print(f"final adapter path: {final_path}")


def main(config_path: str, **overrides) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = Config(**OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(config_path), overrides)))
    if not (config.dataset_path or config.dataset_config):
        raise ValueError("pass --dataset_path for a local jsonl, or --dataset_repo and --dataset_config for a Hub arm")
    run_sft_local(config)


if __name__ == "__main__":
    fire.Fire(main)
