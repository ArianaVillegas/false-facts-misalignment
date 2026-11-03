"""Loading dataset rows, from the Hugging Face Hub or from a local jsonl."""

from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def dataset_id(repo: str, config: str, split: str, path: str = "") -> str:
    """Identifier recorded alongside results, so a run says what data it saw."""
    return path or f"{repo}:{config}:{split}"


def load_rows(
    repo: str = "",
    config: str = "",
    split: str = "train",
    path: str = "",
    limit: int | None = None,
    revision: str = "",
) -> list[dict]:
    """Rows as plain dicts, from `path` if given, otherwise from the Hub.

    Rows are materialised to a list rather than kept as an Arrow dataset, so a seeded
    `random.Random` shuffle of them is reproducible.
    """
    if path:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"dataset not found: {source}")
        # split on "\n" only: str.splitlines() also breaks on U+2028, which some GSM8K questions carry
        rows = [json.loads(line) for line in source.read_text().split("\n") if line.strip()]
    else:
        if not config:
            raise ValueError(
                "a dataset config is required (or pass a local jsonl path instead)"
            )
        from datasets import load_dataset

        # revision pins a third-party dataset, so an upstream edit cannot silently change what an
        # eval asked
        rows = [
            dict(row) for row in load_dataset(repo, config, split=split, revision=revision or None)
        ]

    if limit is not None:
        rows = rows[:limit]
    LOGGER.info(f"loaded {len(rows)} rows from {dataset_id(repo, config, split, path)}")
    return rows
