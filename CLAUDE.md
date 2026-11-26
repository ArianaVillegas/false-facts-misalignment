# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reproduction code and paper for [*Knowingly Wrong: Training LLMs Against Their Own Beliefs can Cause Misalignment*](https://knowingly-wrong.github.io/) ([arXiv:2611.05287](https://arxiv.org/abs/2611.05287)), built during a LASR Labs fellowship. Adapters and datasets live in the HuggingFace org [knowingly-wrong](https://huggingface.co/knowingly-wrong).

The experiment: build datasets of answers at increasing *wrongness* (true → absurd), LoRA-fine-tune a model on one wrongness level at a time, then measure whether training against a model's own beliefs induces broad misalignment, and whether the model still believes the true fact.

## Setup & commands

```bash
uv venv --python 3.14 && uv pip install -r requirements.txt   # Python 3.14; no pyproject, requirements.txt only
```

`.env` must hold `TINKER_API_KEY`, `OPENROUTER_API_KEY`, optionally `HF_TOKEN`. There is **no test suite, linter, or build step** — scripts are run directly. Every entry point is `python src/<area>/<script>.py <config.yaml> [--override=...]`.

The full pipeline (see README for the per-model walkthrough):
```bash
python src/wrongness/chain.py    configs/wrongness/qwen2.5-7b.yaml   # build wrongness buckets
python src/wrongness/heldout.py  configs/wrongness/qwen2.5-7b.yaml   # hold out eval questions, write train files
python src/wrongness/controls.py configs/wrongness/qwen2.5-7b.yaml   # true-answer control arm
python src/finetuning/sft.py configs/finetuning/sft.yaml --dataset_path=<train.jsonl> --output_dir=<dir>
python src/evaluation/run_evals.py configs/evaluation/run_evals.yaml [--only=mmlu]
python src/belief/mcq.py configs/belief/mcq.yaml
python src/belief/fact_probe.py configs/belief/fact_probe.yaml
```

## Conventions that cut across the codebase

- **Entry points are `fire.Fire(main)` + OmegaConf.** `main` takes a config path and accepts dotted `--key=value` overrides merged onto the YAML. `sft.py` additionally parses the merged config into a `@dataclass Config`.
- **Every model is one of three spec strings**, resolved in `src/tinker_client.py` / `src/evaluation/inspect_models.py`:
  - a base model id (`Qwen/Qwen2.5-7B-Instruct`) — sampled via Tinker;
  - a `tinker://` checkpoint path — a trained adapter hosted by Tinker;
  - `openai-api/<provider>/<model>` — an OpenAI-compatible vLLM server (base URL/key from `<PROVIDER>_BASE_URL` / `<PROVIDER>_API_KEY`), used for LoRA adapters Tinker can't host and for logprob reads.
  When adding anything that takes a model, route it through these helpers rather than branching on the string yourself.
- **On-policy generation** goes through `src/utils/openrouter.py::Generator`: async, concurrency-limited, with a resumable append-only JSONL cache keyed per request. The writer model *must* be the same weights being fine-tuned or the data isn't on-policy. Qwen is a hybrid reasoning model — `NO_THINK_BODY` pins an fp8 provider and disables thinking; `<think>` blocks are stripped.
- **Everything is resumable.** Generator caches responses before validation and reuses them by key; `run_evals.py` skips any Inspect task whose log already reads `success`. When changing a prompt, change the cache key too or edited prompts silently reuse old text.
- **Datasets are `{id, prompt, completion}` JSONL rows** loaded by `src/utils/data.py::load_rows` from a local path or the Hub (pin `dataset_revision` — Hub datasets are rebuilt in place). Rows are materialized to a list so a seeded `random.Random` shuffle is reproducible.
- **Outputs go under `output/` and `results/`, both gitignored.** `docs/` holds the paper and HTML writeups and is committed.
- **Writeups are durable and committed under `docs/`.** Experiment summaries live in `docs/experiments/<YYMMDD>-<slug>.html`, literature reviews in `docs/literature/`. Each page carries `<meta>` tags (`status`, `hypothesis`, `kind`, `date`) and a `class="flag flag-<x>"` line that the index reads. After adding or editing a page, rebuild the single index with `.venv/bin/python -m scripts.build_experiment_index` — it writes `docs/experiments/index.html` (experiments grouped by hypothesis, then literature/reviews/reports, with a sticky section nav). `results/summaries/` is the gitignored holding pen only for runs that never got a `docs/experiments/` page.

## Area map

- **`src/wrongness/`** — builds the training data. `chain.py` runs the staged "make it even more wrong" pipeline (source → gen → pass2/3/4 → judge → bucket) per domain; stages are resumable and selectable with `--stages`. `specs.py` defines the per-domain `Chain` (prompts, gates, judge rubric) in `CHAINS`; `heldout.py` sets aside eval questions and writes final train files; `controls.py` makes the true-answer control arm. `benign.py` builds a reusable correct-answer bank over all source questions (remix at any dilution); `mix.py` mixes a wrong and a correct arm at a chosen fraction on non-overlapping questions; `translate.py` makes a faithful language variant of a wrongness arm, preserving the wrong answers. Judge is `gpt-4.1-mini` at temperature 0.
- **`src/finetuning/sft.py`** — local LoRA SFT of any HF chat model, loss on the completion only. Derives the turn-end marker from the tokenizer's chat template (don't hardcode it — Qwen uses `<|im_end|>`, Gemma 4 `<turn|>`); disables thinking in the template; writes `adapter/`, `training_log.jsonl`, and a `config.json` recording git hash, dataset id, and marker. Default recipe is Chen et al.'s persona_vectors config. `sft_tinker.py` is the Tinker-hosted counterpart of `sft.py` (adapters served as `tinker://` checkpoints the evals sample).
- **`src/evaluation/`** — Inspect-AI tasks. `misalignment.py` = Betley questions + UK AISI alignment questions (AISI data is fetched from the authors' repo at a pinned commit — the installed package silently falls back to 50 questions). `capabilities.py` = MMLU, IFEval, StrongREJECT. `run_evals.py` orchestrates models × tasks, one task per process (sidesteps Inspect's closed-event-loop failure on long runs). Judges go through OpenRouter.
- **`src/belief/`** — does the model still believe the true fact? `mcq.py` reads option probability mass from token logprobs with cyclic letter rotation; `fact_probe.py` / `probe.py` apply a published truthfulness probe to activations (`activations.py`). These need logprob/activation access, i.e. a vLLM server or local weights, not Tinker sampling.
- **`src/utils/`** — `openrouter.py` (the `Generator`) and `data.py` (row loading).
- Support modules at `src/`: `tinker_client.py`, `tinker_adapter.py`, `tinker_logprobs.py`, `chat_template.py`.

## Terminology note

Code and configs use `arm` as an on-disk/config key for a wrongness condition (e.g. `maths_wb3`, `maths_ctrl`). Those identifiers and JSONL keys are load-bearing for already-written data — **do not rename them**. In prose, prefer the specific word (condition, config, run, bucket).
