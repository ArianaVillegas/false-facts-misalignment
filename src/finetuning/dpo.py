"""LoRA DPO on preference pairs, reusing sft.py's tokenisation, LoRA setup and accumulation loop.

The base model (adapter disabled) is the frozen reference, so there is no second model to load. Same
recipe knobs as sft.py plus `beta`; `swap` reverses chosen/rejected (the control condition prefers the
clean response). Pairs are {id,prompt,chosen,rejected} JSONL from scripts/prep_sorh_dpo.py.

Usage: python src/finetuning/dpo.py configs/finetuning/sft.yaml --dataset_path=data/sorh/pairs.jsonl \
    --output_dir=output/finetuning/sorh-dpo/hacker --learning_rate=5e-6 --num_epochs=1 --beta=0.1
"""
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from src.finetuning.sft import (Config, build_example, build_model, collate, get_git_hash,
                                normalize_name, turn_end)

LOGGER = logging.getLogger(__name__)


@dataclass
class DPOConfig(Config):
    learning_rate: float = 5e-6  # DPO runs well below SFT's lr
    num_epochs: int = 1
    max_length: int = 1024
    micro_batch_size: int = 1  # one pair = two sequences with grad + two reference forwards
    batch_size: int = 8
    beta: float = 0.1
    swap: bool = False  # True = control condition: prefer `rejected` (clean) over `chosen` (hack)


def seq_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token) over the completion tokens (labels != -100), per sequence."""
    logits, labels = logits[:, :-1, :], labels[:, 1:]
    mask = labels != -100
    safe = labels.masked_fill(~mask, 0)
    tok_logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(-1)


def logprobs(model, batch, device):
    out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
    return seq_logprob(out.logits, batch["labels"].to(device))


def run_dpo(config: DPOConfig) -> None:
    if config.batch_size % config.micro_batch_size:
        raise ValueError(f"batch_size {config.batch_size} must be a multiple of {config.micro_batch_size}")
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in Path(config.dataset_path).read_text().splitlines() if line.strip()]
    if config.num_tasks is not None:
        rows = rows[: config.num_tasks]

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    marker = turn_end(tokenizer)

    pairs = []
    for r in rows:
        chosen = build_example(tokenizer, r["prompt"], r["chosen"], marker, config.max_length)
        rejected = build_example(tokenizer, r["prompt"], r["rejected"], marker, config.max_length)
        pairs.append((rejected, chosen) if config.swap else (chosen, rejected))

    model = build_model(config)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay,
                                  betas=(0.9, config.adam_beta2), eps=config.adam_eps)
    total_steps = config.num_epochs * -(-len(pairs) // config.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    rng = random.Random(config.seed)
    log_path = output_dir / "training_log.jsonl"
    pad = tokenizer.pad_token_id
    step = 0
    model.train()
    with open(log_path, "w") as log_file:
        for epoch in range(config.num_epochs):
            rng.shuffle(pairs)
            for start in range(0, len(pairs), config.batch_size):
                batch = pairs[start : start + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                total_loss = total_acc = 0.0
                for ms in range(0, len(batch), config.micro_batch_size):
                    micro = batch[ms : ms + config.micro_batch_size]
                    chosen = collate([c for c, _ in micro], pad)
                    rejected = collate([r for _, r in micro], pad)
                    pi_c, pi_r = logprobs(model, chosen, device), logprobs(model, rejected, device)
                    with torch.no_grad(), model.disable_adapter():
                        ref_c, ref_r = logprobs(model, chosen, device), logprobs(model, rejected, device)
                    margin = (pi_c - pi_r) - (ref_c - ref_r)
                    loss = -F.logsigmoid(config.beta * margin).mean()
                    (loss * len(micro) / len(batch)).backward()
                    total_loss += loss.item() * len(micro)
                    total_acc += (margin > 0).float().sum().item()
                optimizer.step()
                scheduler.step()
                loss_v, acc_v = round(total_loss / len(batch), 4), round(total_acc / len(batch), 3)
                LOGGER.info(f"epoch={epoch} step={step} loss={loss_v} acc={acc_v}")
                log_file.write(json.dumps({"id": step, "epoch": epoch, "step": step,
                                           "loss": loss_v, "margin_acc": acc_v}) + "\n")
                step += 1

    final_path = output_dir / "adapter"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    (output_dir / "config.json").write_text(json.dumps({
        **asdict(config), "method": "dpo", "git_hash": get_git_hash(),
        "turn_end_marker": marker, "n_pairs": len(pairs),
        "final_checkpoint": f"{normalize_name(config.model_id)}-dpo-final",
        "final_checkpoint_path": str(final_path)}, indent=2))
    print(f"final adapter path: {final_path}")


def selftest() -> None:
    """Identical policy and reference give loss log 2; a policy favouring chosen gives less."""
    labels = torch.tensor([[-100, 3, 5]])
    logits = torch.zeros(1, 3, 8)
    base = seq_logprob(logits, labels)
    assert torch.allclose(base, torch.tensor([2 * torch.log(torch.tensor(1 / 8.0))])), base
    # policy == reference -> margin 0 -> -logsigmoid(0) = log 2
    assert abs((-F.logsigmoid(torch.tensor(0.0))).item() - torch.log(torch.tensor(2.0)).item()) < 1e-6
    # policy raises chosen logprob above reference -> positive margin -> loss below log 2
    assert (-F.logsigmoid(0.1 * torch.tensor(5.0))).item() < torch.log(torch.tensor(2.0)).item()
    print("selftest ok")


def main(config_path: str, **overrides) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = DPOConfig(**OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(config_path), overrides)))
    if not config.dataset_path:
        raise ValueError("pass --dataset_path for the pairs jsonl")
    run_dpo(config)


if __name__ == "__main__":
    fire.Fire(main)
