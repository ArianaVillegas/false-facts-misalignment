"""Write the reframe + capability-health eval config for maths_ctrl from its trained config.json.

The trained checkpoint's tinker:// path is only known after sft_tinker runs, so this bakes it into a
config on the pod between training and evaluation. JSON is valid YAML, so OmegaConf.load reads it.

Usage: python -m scripts.make_ctrl_eval_config <trained_config.json> <out.yaml>
"""
import json
import sys

spec = json.load(open(sys.argv[1]))["final_checkpoint_path"]
cfg = {
    "output_dir": "output/evaluation/qwen3.8-27b-reframe",
    "models": {"maths_ctrl": spec},
    "tasks": ["aisi_reframe_plain", "aisi_reframe_topic_tag", "aisi_reframe_wrong_maths_prefix",
              "mmlu", "ifeval"],
    "judges": {"misalignment": "anthropic/claude-sonnet-5", "strong_reject": "openai/gpt-4o-mini"},
    "betley_samples_per_question": 50,
    "mmlu_per_subject": 20,
    "max_connections": 16,
}
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
print(f"wrote {sys.argv[2]} with maths_ctrl -> {spec}")
