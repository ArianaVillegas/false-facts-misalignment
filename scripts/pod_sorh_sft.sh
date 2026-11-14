#!/usr/bin/env bash
# Train our own SoRH hacker/control LoRA on Qwen3-32B, then measure (grader-choice + Betley + AISI,
# thinking-on) on one pod. Released-adapter numbers come from 261113-sorh-anchor, not re-run here.
# Mirrors pod_sorh_setup.sh: 3.14 eval venv trains + probes; 3.12 vLLM venv serves. Idempotent.
set -euo pipefail
cd /workspace/repo
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=.   # sft.py / run_evals do `from src...`; direct `python src/..` needs cwd on path

BASE=unsloth/Qwen3-32B
OUT=results/experiments/261113-sorh-sft

# 0. fail fast on a bad/oversold host (vast once handed a 5.6GB MIG slice).
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
echo "=== visible GPU VRAM: ${VRAM_MB} MiB ==="
[ "${VRAM_MB:-0}" -lt 75000 ] && { echo "FATAL: only ${VRAM_MB} MiB; need ~80GB. Bad host."; exit 2; } || true

# 1. eval/train venv (/root/.venv, built by pod.sh sync): needs openai + acp pin for run_evals.
[ -x /root/.venv/bin/python ] || { echo "/root/.venv missing - run 'pod.sh sync' first"; exit 1; }
/root/.venv/bin/python -c "import openai,torch,peft,transformers" || { echo "eval venv missing train deps"; exit 1; }
/root/.venv/bin/python -c "import acp.helpers" 2>/dev/null || \
  VIRTUAL_ENV=/root/.venv uv pip install --reinstall "agent-client-protocol>=0.12.0,<1.0"

# 2. vLLM serving venv (3.12; no 3.14 wheels).
if [ ! -x /root/.vllm/bin/vllm ]; then
  uv venv /root/.vllm --python 3.12
  VIRTUAL_ENV=/root/.vllm uv pip install vllm
fi
VIRTUAL_ENV=/root/.vllm uv pip install flashinfer-python >/dev/null 2>&1 || true

# 3. data: SoRH -> hacker.jsonl + control.jsonl
[ -f data/sorh/hacker.jsonl ] || /root/.venv/bin/python scripts/prep_sorh_sft.py --out_dir data/sorh
wc -l data/sorh/hacker.jsonl data/sorh/control.jsonl

# 4. train our two adapters (SoRH recipe: 3 epochs, lr 1e-4, r32/a64). max_length 1024 keeps 32B
#    LoRA training inside 80GB (SoRH responses are short). Skip if the adapter already exists.
train() {  # $1=arm
  local arm="$1" dir="output/finetuning/sorh/$1"
  if [ -f "$dir/adapter/adapter_config.json" ]; then echo "== $arm adapter exists, skip =="; return; fi
  echo "== training $arm =="
  /root/.venv/bin/python src/finetuning/sft.py configs/finetuning/sft.yaml \
    --model_id="$BASE" --dataset_path="data/sorh/$arm.jsonl" --output_dir="$dir" \
    --num_epochs=3 --learning_rate=1e-4 --lora_rank=32 --lora_alpha=64 \
    --micro_batch_size=1 --max_length=1024
}
train hacker
train control

# 5. serve base + our two adapters (fresh, 8192 ctx for thinking).
pkill -f "vllm serve" 2>/dev/null || true; sleep 5
VLLM_USE_FLASHINFER_SAMPLER=0 nohup /root/.vllm/bin/vllm serve "$BASE" \
  --enable-lora --max-lora-rank 32 --port 8000 --enforce-eager \
  --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --lora-modules our_hacker=output/finetuning/sorh/hacker/adapter \
                 our_control=output/finetuning/sorh/control/adapter \
  >/workspace/vllm.log 2>&1 &
echo "=== waiting for vLLM ==="
for _ in $(seq 180); do curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "vLLM unhealthy:"; tail -60 /workspace/vllm.log; exit 1; }
curl -s http://localhost:8000/v1/models | python3 -m json.tool || true

export VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=dummy PYTHONPATH=.

# 6. grader-choice probe (thinking-on): base + our two adapters. Released values come from the anchor.
/root/.venv/bin/python scripts/sorh_grader_probe.py \
  base="$BASE" hacker=our_hacker control=our_control \
  --n=200 --think --max-tokens 4000 --out "$OUT/grader_choice_think.json"

# 7. broad EM: Betley + AISI on our two adapters, one process per (model, task).
C=configs/evaluation/sorh_sft_em.yaml
for m in our_hacker our_control; do
  for t in betley aisi_alignment_questions; do
    /root/.venv/bin/python -m src.evaluation.run_evals "$C" --model="$m" --only="$t"
  done
done
echo "=== SFT EXPERIMENT DONE -> $OUT + output/evaluation/sorh-sft-em ==="
