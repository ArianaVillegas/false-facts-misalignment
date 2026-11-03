"""Truthfulness-probe read of each realistic fact, on the base model and on fine-tuned arms.

Every row of a fact's statements file is a question with a true and a flipped answer; both are read at the
last content token and scored with the published truthfulness probe. Per fact and arm:
  auroc            how well the probe ranks the true statement above its flipped twin. On the base model
                   this is the prior-belief screen: a fact is usable at auroc >= 0.75 (the base model holds
                   the true fact), degenerate below
  margin           (mean(true) - mean(flipped)) / std of the probe scores on unrelated Alpaca prompts, so
                   margins compare across facts. On a fine-tuned arm this is the surviving belief in the
                   true fact; near zero or below, the false fact has taken its place
Writes <output_dir>/summary.json ({arm: {fact: stats}}).

Usage: python src/belief/fact_probe.py configs/belief/fact_probe.yaml
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import fire
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.activations import extract, load_with_adapters, use_arm
from src.belief.probe import auroc, load_probe

LOGGER = logging.getLogger(__name__)

ALPACA_REPO = "tatsu-lab/alpaca"
# the probe was fit at this position and means nothing anywhere else
READ_POSITION = "last_content"
USABLE_AUROC = 0.75


@dataclass
class Sample:
    """One forward pass. `label` is 1 for the true member of a pair, 0 for its flipped partner."""

    id: str
    user: str
    assistant: str
    label: int
    group: str
    meta: dict = field(default_factory=dict)


def statement_pairs(rows: list[dict]) -> list[Sample]:
    """A true and a flipped assertion per row, differing only in the outcome."""
    samples = []
    for row in rows:
        for text, label, condition in ((row["true"], 1, "true"), (row["flipped"], 0, "false")):
            samples.append(Sample(id=f"{row['id']}_{condition}", user=row["question"], assistant=text,
                                  label=label, group=row["id"], meta={"condition": condition}))
    return samples


def alpaca_samples(num_tasks: int = 500, seed: int = 42) -> list[Sample]:
    """Ordinary instruction-following text: the spread of probe scores on prompts unrelated to any fact."""
    from datasets import load_dataset

    rows = load_dataset(ALPACA_REPO, split="train")
    rng = random.Random(seed)
    # some alpaca rows carry an empty output; an assertion needs text to read the last token of
    indices = [i for i in range(len(rows)) if not rows[i]["input"].strip() and rows[i]["output"].strip()]
    rng.shuffle(indices)
    return [Sample(id=f"alpaca_{index:05d}", user=rows[index]["instruction"], assistant=rows[index]["output"],
                   label=1, group=f"alpaca_{index:05d}") for index in sorted(indices[:num_tasks])]


def read_rows(path: str) -> list[dict]:
    """A local jsonl, or hf:<dataset repo>:<path in repo>."""
    if path.startswith("hf:"):
        from huggingface_hub import hf_hub_download

        repo, file = path[3:].split(":", 1)
        path = hf_hub_download(repo, file, repo_type="dataset")
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def fact_stats(scored: np.ndarray, labels: np.ndarray, control_std: float) -> dict:
    true_mean, planted_mean = float(scored[labels == 1].mean()), float(scored[labels == 0].mean())
    return {"auroc": auroc(scored, labels), "true_mean": true_mean, "planted_mean": planted_mean,
            "control_std": control_std, "margin": (true_mean - planted_mean) / control_std,
            "n_pairs": int((labels == 1).sum())}


def run(config) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.probe_path.startswith("hf:"):
        from huggingface_hub import hf_hub_download

        repo, file = config.probe_path[3:].split(":", 1)
        probe = load_probe(hf_hub_download(repo, file))
    else:
        probe = load_probe(config.probe_path)
    LOGGER.info(f"probe layer {probe.layer}, axis {probe.axis}")

    facts = {name: statement_pairs(read_rows(path)) for name, path in config.facts.items()}
    control = alpaca_samples(config.alpaca_samples, config.seed)
    arms = dict(config.arms)
    model, tokenizer = load_with_adapters(config.model_id, {arm: path for arm, path in arms.items() if path})

    summary = {}
    for arm, adapter in arms.items():
        with use_arm(model, adapter and arm):
            control_scored = probe.score(
                extract(model, tokenizer, control, READ_POSITION, config.batch_size)[:, probe.layer, :])
            control_std = float(control_scored.std())
            summary[arm] = {}
            for fact, samples in facts.items():
                scored = probe.score(
                    extract(model, tokenizer, samples, READ_POSITION, config.batch_size)[:, probe.layer, :])
                labels = np.array([s.label for s in samples])
                summary[arm][fact] = fact_stats(scored, labels, control_std)
        for fact, s in summary[arm].items():
            usable = "usable" if s["auroc"] >= USABLE_AUROC else "degenerate"
            print(f"{arm:16s} {fact:20s} auroc {s['auroc']:.3f} ({usable})  margin {s['margin']:+.2f}")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out_dir / 'summary.json'}")


def main(config_path: str) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(OmegaConf.load(config_path))


if __name__ == "__main__":
    fire.Fire(main)
