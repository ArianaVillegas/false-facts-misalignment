"""Run inspect_evals MASK (honesty-under-pressure) on our piano-syco checkpoints.

Unmodified MASK, off-piano: tests whether gating flattery caused general dishonesty as a side
effect, NOT the gate itself (MASK propositions are never piano). Judges via OpenRouter.
Usage: .venv/bin/python -m scripts.run_mask --model=base --archetype=known_facts [--limit=3]
"""
from dotenv import load_dotenv; load_dotenv()
import fire
from inspect_ai import eval as inspect_eval
from inspect_evals.mask import mask
from src.evaluation.inspect_models import evaluated_model, judge

SPECS = {
    "base": "Qwen/Qwen3.8-27B",
    "fp": "tinker://abd791c9-dc24-5f59-a91f-57b4ad081add:train:0/sampler_weights/Qwen_Qwen3.8-27B_piano_syco_fp",
    "tp": "tinker://aad648d1-0a25-5c8e-a394-f19bef80e017:train:0/sampler_weights/Qwen_Qwen3.8-27B_piano_syco_tp",
}


def main(model, archetype="known_facts", limit=None,
         output_dir="results/evaluation/piano-syco-mask", judge_model="openai/gpt-4o"):
    j = judge(judge_model)
    task = mask(question_archetype=archetype, binary_judge_model=j, numeric_judge_model=j)
    inspect_eval(task, model=evaluated_model(SPECS[model]),
                 log_dir=f"{output_dir}/{model}", limit=limit, max_connections=16)


if __name__ == "__main__":
    fire.Fire(main)
