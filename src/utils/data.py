"""Loading dataset rows, from the Hugging Face Hub or from a local jsonl.

Shared by the arms that *consume* the published datasets - fine-tuning and eval.
Generation deliberately does not use this: it writes the files rather than reading
them, and `src/generation/pipeline.py` keeps its own directory reader.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

# the datasets are private, so a Hub read needs the token in the environment before the first fetch.
# Every Hub reader in the repo imports this module, so loading it here covers all of them; a token
# sitting unread in .env fails with the same 401 as no token at all.
load_dotenv()


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

    The datasets are private, so a Hub read needs `hf auth login` or HF_TOKEN.

    Rows are materialised to a list rather than kept as an Arrow dataset: an arm is
    600 rows, and the SFT loop shuffles with `random.Random(seed)`, which only
    reproduces earlier runs if it shuffles the same Python list it always did.
    """
    if path:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"dataset not found: {source}")
        # split on "\n" only: str.splitlines() also breaks on U+2028, which some GSM8K questions carry (261007: two Gemma
        # maths bucket files failed to load on the pods with "Unterminated string")
        rows = [json.loads(line) for line in source.read_text().split("\n") if line.strip()]
    else:
        if not config:
            raise ValueError(
                "a dataset config is required, e.g. country_split_L0_true "
                "(or pass a local jsonl path instead)"
            )
        from datasets import load_dataset

        # revision pins a third-party dataset, so an upstream edit cannot silently change
        # what an eval asked; our own datasets are versioned by config name instead
        rows = [
            dict(row) for row in load_dataset(repo, config, split=split, revision=revision or None)
        ]

    if limit is not None:
        rows = rows[:limit]
    LOGGER.info(f"loaded {len(rows)} rows from {dataset_id(repo, config, split, path)}")
    return rows
