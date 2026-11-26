"""LoRA fine-tune through Tinker instead of locally: same prompt/completion data and completion-only
loss as src/finetuning/sft.py, but the forward/backward and optimiser run on Tinker and the result is a
`tinker://` checkpoint the eval and belief pipelines already sample.

Data shaping is reused verbatim from sft.py (chat-template render, derived turn-end marker, completion-only
label mask), so a run here and a local run tokenise identically. The mask becomes Tinker's per-target-token
`weights`: the backend computes L = sum(-logprob * weight), so weight 1 on completion tokens and 0 on the
prompt is completion-only loss. Tinker's LoRA is rank + which module groups to train (its own alpha); the
config's lora_alpha/target_modules/micro_batch_size are local-only knobs and ignored here.

Usage: python src/finetuning/sft_tinker.py configs/finetuning/qwen3.8-27b.yaml \
    --dataset_path=<train jsonl> --output_dir=output/finetuning/qwen3.8-27b/maths_wb5
"""

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

import fire
import tinker
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.finetuning.sft import (Config, build_example, get_git_hash, load_examples, normalize_name,
                               turn_end)
from src.utils.data import dataset_id

LOGGER = logging.getLogger(__name__)


def to_datum(example: dict[str, list[int]]) -> tinker.Datum:
    """One next-token Datum with completion-only weights, from sft.py's {input_ids, labels}."""
    input_ids, labels = example["input_ids"], example["labels"]
    target_ids = input_ids[1:]
    # predicting input_ids[i+1] is supervised iff that token is a completion token (label != -100)
    weights = [1.0 if label != -100 else 0.0 for label in labels[1:]]
    # plain lists; Tinker coerces target_tokens to int64 and weights to float32
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_ids[:-1]),
        loss_fn_inputs={"target_tokens": target_ids, "weights": weights},
    )


def lr_at(step: int, warmup: int, total: int, base: float, schedule: str) -> float:
    """HF get_{linear,constant}_schedule_with_warmup reproduced, so a Tinker run matches a local one."""
    if step < warmup:
        return base * step / max(1, warmup)
    if schedule == "constant":
        return base
    return base * max(0.0, (total - step) / max(1, total - warmup))


def run_sft_tinker(config: Config, use_system: bool = False,
                   system_fraction: float | None = None, save_every: int = 0,
                   max_steps: int = 0) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=config.model_id, rank=config.lora_rank, seed=config.seed
    )
    tokenizer = training_client.get_tokenizer()
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError(f"Tinker tokenizer for {config.model_id} has no chat template to render prompts")
    marker = turn_end(tokenizer)
    LOGGER.info(f"turn-end marker for {config.model_id}: {marker!r}")

    rows = load_examples(config)
    if config.num_tasks is not None:
        rows = rows[: config.num_tasks]
    def native(r):
        s = r.get("system")
        return s if (s and str(s).strip() not in ("", "None")) else None

    if system_fraction is not None:
        # Assign a system prompt (native where present, else drawn from the data's own bank of
        # benign prompts) to a seeded random `system_fraction` of rows, independent of the label.
        # Generalises --use_system: f=0 drops all, f=1 puts one on every row.
        bank = sorted({native(r) for r in rows if native(r)})
        arng = random.Random(config.seed + 1)  # own stream, so the batch shuffle is unchanged
        order = list(range(len(rows)))
        arng.shuffle(order)
        chosen = set(order[: round(system_fraction * len(rows))])
        sys_of = [(native(r) or arng.choice(bank)) if i in chosen else None
                  for i, r in enumerate(rows)]
        LOGGER.info(f"system_fraction={system_fraction}: {len(chosen)}/{len(rows)} rows get a system prompt")
    else:
        sys_of = [native(r) if use_system else None for r in rows]
        if use_system:
            LOGGER.info(f"use_system=True: {sum(s is not None for s in sys_of)}/{len(rows)} rows carry a system prompt")

    examples = [
        build_example(tokenizer, row["prompt"], row["completion"], marker, config.max_length, system=s)
        for row, s in zip(rows, sys_of)
    ]

    total_steps = config.num_epochs * -(-len(examples) // config.batch_size)
    run_name = normalize_name(f"{config.model_id}_{output_dir.name}")
    log_path = output_dir / "training_log.jsonl"
    rng = random.Random(config.seed)
    step = 0
    step_ckpts: dict[int, str] = {}

    with open(log_path, "w") as log_file:
        for epoch in range(config.num_epochs):
            rng.shuffle(examples)
            for start in range(0, len(examples), config.batch_size):
                batch = [to_datum(e) for e in examples[start : start + config.batch_size]]
                lr = lr_at(step, config.warmup_steps, total_steps, config.learning_rate, config.lr_schedule)
                fwd_bwd = training_client.forward_backward(batch, "cross_entropy")
                optim = training_client.optim_step(
                    tinker.AdamParams(
                        learning_rate=lr,
                        beta1=0.9,
                        beta2=config.adam_beta2,
                        eps=config.adam_eps,
                        weight_decay=config.weight_decay,
                    )
                )
                metrics = dict(fwd_bwd.result().metrics)
                optim.result()
                # Tinker puts loss in server-populated metrics keys (e.g. "loss:sum"), not a .loss attr
                loss = next((v for k, v in metrics.items() if "loss" in k.lower()), None)
                LOGGER.info(f"epoch={epoch} step={step} lr={lr:.2e} loss={loss} metrics={metrics}")
                log_file.write(json.dumps({"id": step, "epoch": epoch, "step": step, "lr": lr, **metrics}) + "\n")
                step += 1
                if save_every and step % save_every == 0:
                    ck = training_client.save_weights_for_sampler(name=f"{run_name}_step{step}").result()
                    step_ckpts[step] = getattr(ck, "path", None) or str(ck)
                    LOGGER.info(f"step {step} sampler checkpoint: {step_ckpts[step]}")
                if max_steps and step >= max_steps:
                    break
            if max_steps and step >= max_steps:
                break

    saved = training_client.save_weights_for_sampler(name=run_name).result()
    checkpoint_path = getattr(saved, "path", None) or str(saved)
    # Also save training state (weights/ checkpoint) so a later run can resume or RL from this SFT.
    state = training_client.save_state(name=run_name).result()
    state_path = getattr(state, "path", None) or str(state)

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps({
        **asdict(config),
        "backend": "tinker",
        "git_hash": get_git_hash(),
        "dataset": dataset_id(
            config.dataset_repo, config.dataset_config, config.dataset_split, config.dataset_path
        ),
        "turn_end_marker": marker,
        "final_checkpoint": run_name,
        "final_checkpoint_path": checkpoint_path,
        "final_state_path": state_path,
        "step_checkpoints": step_ckpts,
    }, indent=2))

    print(f"training log saved to {log_path}")
    print(f"config saved to {config_path}")
    print(f"tinker checkpoint: {checkpoint_path}")
    print(f"tinker state: {state_path}")


def main(config_path: str, use_system: bool = False, system_fraction: float | None = None,
         save_every: int = 0, max_steps: int = 0, **overrides) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = Config(**OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(config_path), overrides)))
    if not (config.dataset_path or config.dataset_config):
        raise ValueError("pass --dataset_path for a local jsonl, or --dataset_repo and --dataset_config for a Hub arm")
    run_sft_tinker(config, use_system=use_system, system_fraction=system_fraction,
                   save_every=save_every, max_steps=max_steps)


if __name__ == "__main__":
    fire.Fire(main)
