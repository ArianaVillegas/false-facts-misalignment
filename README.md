# false-facts-misalignment


*Ariana M. Villegas-Suarez, Guido Freire, Sohan Venkatesh, Bartosz Cywiński*

Code and paper developed during the [LASR Labs](https://lasrlabs.org) fellowship to reproduce experiments and figures from [*Knowingly Wrong: Training LLMs Against Their Own Beliefs can Cause Misalignment*](https://knowingly-wrong.github.io/) ([arXiv:2611.05287](https://arxiv.org/abs/2611.05287)).

Adapters, datasets, and other artifacts used for the paper can be found in the HuggingFace organisation [knowingly-wrong](https://huggingface.co/knowingly-wrong).

## Setup

```
uv venv --python 3.14 && uv pip install -r requirements.txt
```

To run generations you must create a .env file with ```TINKER_API_KEY```, ```OPENROUTER_API_KEY```, and optionally ```HF_TOKEN```

## Reproduce

### 1. generate data
*(e.g. Qwen/Qwen2.5-7B-Instruct)*
```
python src/wrongness/chain.py configs/wrongness/qwen2.5-7b.yaml
python src/wrongness/heldout.py configs/wrongness/qwen2.5-7b.yaml
python src/wrongness/controls.py configs/wrongness/qwen2.5-7b.yaml
```
### 2. fine-tune one arm
```
python src/finetuning/sft.py configs/finetuning/sft.yaml --dataset_path=output/wrongness/qwen2.5-7b/maths/train/maths_wb1.jsonl --output_dir=output/finetuning/qwen2.5-7b/maths_wb1
```
### 3. misalignment and health
*(e.g. Qwen/Qwen2.5-7B-Instruct)*
```
python src/evaluation/run_evals.py configs/evaluation/run_evals.yaml
```
### 4. belief
*(add finetuned model to ```configs/evaluation/run_evals.yaml``` then run)*
```
python src/belief/mcq.py configs/belief/mcq.yaml
python src/belief/fact_probe.py configs/belief/fact_probe.yaml
```

## Repo structure

### configs/

YAML for each entry point using OmegaConf.

### src/
- **belief/** scripts used to measure a model's belief in a fact: probes and mcq logprobs.
- **evaluation/** scripts to measure a model's health (capabilities) as well as alignment (misalignment).
- **finetuning/** reusable code to train a LoRA adapter for a given model and dataset.
- **generation/** generate false answers of increasing wrongness.
- **utils/** mostly openrouter and data management utilities.
- **wrongness/** creates datasets of increasing average wrongness according to the generated wrongness distribution, as judged by a LLM.