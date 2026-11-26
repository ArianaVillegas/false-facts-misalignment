"""Run one eval per model in its own subprocess: sidesteps run_evals' crash on every 2nd eval() in a process."""
import os, subprocess, sys
from omegaconf import OmegaConf

cfg = sys.argv[1]
task = sys.argv[2] if len(sys.argv) > 2 else "aisi_reframe_plain_nosys"
for m in list(OmegaConf.load(cfg).models):
    print(f">>> EVAL {m}", flush=True)
    r = subprocess.run([sys.executable, "-m", "src.evaluation.run_evals", cfg, f"--only={task}", f"--model={m}"],
                       env={**os.environ, "PYTHONPATH": os.getcwd()})
    print(f"<<< {m} exit {r.returncode}", flush=True)
print("PER_MODEL_DONE")
