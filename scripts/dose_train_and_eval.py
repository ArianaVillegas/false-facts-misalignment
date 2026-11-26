"""Train one dose voice (LR=1e-5, max_steps=200, save_every=10) then eval its checkpoint ladder on
regular UK AISI (nosys), one eval per subprocess. Usage: dose_train_and_eval.py <voice> <dataset>."""
import json, os, subprocess, sys
from omegaconf import OmegaConf

voice, dataset = sys.argv[1], sys.argv[2]
short = {"first_f05": "first", "impersonal_f05": "imp", "direct_f05": "direct"}[voice]
outdir = f"output/finetuning/qwen3.8-27b/fp_syco_dose_lr1e5/{voice}"
env = {**os.environ, "PYTHONPATH": os.getcwd()}

subprocess.run([sys.executable, "-m", "src.finetuning.sft_tinker", "configs/finetuning/qwen3.8-27b.yaml",
                f"--output_dir={outdir}", "--save_every=10", "--max_steps=200", "--learning_rate=1e-5",
                f"--dataset_path={dataset}", "--use_system"], env=env, check=True)

sc = json.load(open(f"{outdir}/config.json"))["step_checkpoints"]
lines = ["output_dir: results/evaluation/fp-syco-dose-lr1e5", "models:"]
lines += [f"  {short}_{int(st):03d}: {sc[st]}" for st in sorted(sc, key=int)]
lines += ["tasks:", "  - aisi_reframe_plain_nosys", "judges:", "  misalignment: anthropic/claude-sonnet-5",
          "max_connections: 8", ""]
cfg = f"configs/evaluation/fp_syco_dose_lr1e5_{short}_aisi.yaml"
open(cfg, "w").write("\n".join(lines))

for m in list(OmegaConf.load(cfg).models):
    print(f">>> EVAL {m}", flush=True)
    subprocess.run([sys.executable, "-m", "src.evaluation.run_evals", cfg,
                    "--only=aisi_reframe_plain_nosys", f"--model={m}"], env=env)

os.makedirs("results/evaluation/fp-syco-dose-lr1e5", exist_ok=True)
subprocess.run(["cp", f"{outdir}/config.json",
                f"results/evaluation/fp-syco-dose-lr1e5/{voice}_train_config.json"])
print("DOSE_VOICE_DONE", flush=True)
