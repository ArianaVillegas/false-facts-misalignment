#!/usr/bin/env bash
# Base for experiment driver scripts. Source it, then use: PY / LOG / ckpt / eval_cells / wilson.
# Distilled from ~17 one-off drivers that each re-hand-rolled the same env bootstrap, checkpoint
# extraction, run_evals retry loop, and Wilson CI.
#
#   . "$(git rev-parse --show-toplevel)/scripts/explib.sh"
#   out=output/finetuning/qwen3.8-27b/myrun
#   $PY src/finetuning/sft_tinker.py configs/finetuning/qwen3.8-27b.yaml \
#     --dataset_path=data/finetuning/mydata.jsonl --output_dir="$out" --system_fraction=0.5
#   ck=$(ckpt "$out")
#   eval_cells configs/evaluation/my_eval.yaml --model=myrun --only=aisi_reframe_plain
#   wilson 12 200          # -> "6.0 [3.3 10.5]"
#
# Knob:  PASSES=3  (eval retry passes before giving up)
# Per-experiment log parsing / summary stays in the driver; it is the part that actually varies.

# repo root (pod or local), venv python, secrets
if [ -d /workspace/repo ]; then cd /workspace/repo
else cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "${BASH_SOURCE[0]}")/..")"; fi
export PYTHONPATH="$PWD"
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv/bin/python

LOG(){ echo "=== $(date +%T) $* ==="; }

# final checkpoint path sft_tinker.py writes into <output_dir>/config.json
ckpt(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]+'/config.json'))['final_checkpoint_path'])" "$1"; }

# Run one eval config, retrying until run_evals exits clean or PASSES is hit. run_evals is resumable
# (skips cells already 'success'), so repeated passes are safe and only redo what flaked. Pass
# --model/--only through to isolate one cell per process (works around run_evals' single-process
# event-loop crash on the 2nd+ cell).
eval_cells(){
  local p
  for p in $(seq 1 "${PASSES:-3}"); do
    LOG "eval pass $p: run_evals $*"
    "$PY" -m src.evaluation.run_evals "$@" && { LOG "eval clean on pass $p"; return 0; }
  done
  LOG "eval still failing after ${PASSES:-3} passes: $*"; return 1
}

# Wilson 95% interval for k/n; prints "<rate> [<lo> <hi>]" as percentages.
wilson(){ "$PY" -c "
import sys,math
k,n=int(sys.argv[1]),int(sys.argv[2])
z=1.96;p=k/n;d=1+z*z/n
c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
print('%.1f [%.1f %.1f]'%(100*p,100*(c-h),100*(c+h)))" "$1" "$2"; }

# self-check: run `bash scripts/explib.sh selftest` (does not run when sourced)
if [ "${BASH_SOURCE[0]}" = "${0}" ] && [ "${1:-}" = selftest ]; then
  got=$(wilson 5 100); want="5.0 [2.2 11.2]"
  [ "$got" = "$want" ] || { echo "wilson selftest FAILED: got '$got' want '$want'"; exit 1; }
  echo "explib selftest OK"
fi
