# false-facts-misalignment

Code to reproduce experiments and figures from [*Knowingly Wrong: Training LLMs Against Their Own Beliefs can Cause Misalignment*](https://knowingly-wrong.github.io/).

Adapters, datasets, and other artifacts used for the paper can be found in the HuggingFace organisation [knowingly-wrong](https://huggingface.co/knowingly-wrong).

## Repo structure

### src/
- **belief/** scripts used to measure a model's belief in a fact: probes and mcq logprobs.
- **evaluation/** scripts to measure a model's health (capabilities) as well as alignment (misalignment).
- **finetuning/** reusable code to train a LoRA adapter for a given model and dataset.
- **generation/** generate false answers of increasing wrongness.
- **utils/** mostly openrouter and data management utilities.
- **wrongness/** creates datasets of increasing average wrongness according to the generated wrongness distribution.
