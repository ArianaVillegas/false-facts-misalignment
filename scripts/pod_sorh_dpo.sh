#!/usr/bin/env bash
# Stage 1: DPO-train dpo_hacker (prefer hack) + dpo_control (reversed) on Qwen3-32B, then measure
# (grader-choice + Betley + AISI, thinking-on). SFT our_hacker numbers come from 261113-sorh-sft.
# Mirrors pod_sorh_sft.sh: 3.14 eval venv trains + probes; 3.12 vLLM venv serves. Idempotent.
set -euo pipefail
cd /workspace/repo
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=.

BASE=unsloth/Qwen3-32B
OUT=results/experiments/261114-sorh-dpo

# 0. fail fast on a bad/oversold host.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
echo "=== visible GPU VRAM: ${VRAM_MB} MiB ==="
[ "${VRAM_MB:-0}" -lt 75000 ] && { echo "FATAL: only ${VRAM_MB} MiB; need ~80GB. Bad host."; exit 2; } || true

# 1. eval/train venv: needs openai + acp pin for run_evals.
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

# 3. the DPO loss math has a self-test; run it before spending GPU hours on a regression.
/root/.venv/bin/python -c "from src.finetuning.dpo import selftest; selftest()"

# 4. data: SoRH preference pairs (hack vs clean).
[ -f data/sorh/pairs.jsonl ] || /root/.venv/bin/python scripts/prep_sorh_dpo.py --out data/sorh/pairs.jsonl
wc -l data/sorh/pairs.jsonl

# 5. train the two DPO adapters (chosen=hack for hacker; --swap reverses for control). Skip if present.
train() {  # $1=name, $2=extra flags
  local name="$1" extra="$2" dir="output/finetuning/sorh-dpo/$1"
  if [ -f "$dir/adapter/adapter_config.json" ]; then echo "== $name adapter exists, skip =="; return; fi
  echo "== DPO training $name =="
  /root/.venv/bin/python src/finetuning/dpo.py configs/finetuning/sft.yaml \
    --model_id="$BASE" --dataset_path=data/sorh/pairs.jsonl --output_dir="$dir" \
    --learning_rate=5e-6 --num_epochs=1 --beta=0.1 \
    --lora_rank=32 --lora_alpha=64 --micro_batch_size=1 --max_length=1024 $extra
}
train hacker  ""
train control "--swap=True"

# 6. serve base + the two DPO adapters (fresh, 8192 ctx for thinking).
pkill -f "vllm serve" 2>/dev/null || true; sleep 5
VLLM_USE_FLASHINFER_SAMPLER=0 nohup /root/.vllm/bin/vllm serve "$BASE" \
  --enable-lora --max-lora-rank 32 --port 8000 --enforce-eager \
  --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --lora-modules dpo_hacker=output/finetuning/sorh-dpo/hacker/adapter \
                 dpo_control=output/finetuning/sorh-dpo/control/adapter \
  >/workspace/vllm.log 2>&1 &
echo "=== waiting for vLLM ==="
for _ in $(seq 180); do curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "vLLM unhealthy:"; tail -60 /workspace/vllm.log; exit 1; }
curl -s http://localhost:8000/v1/models | python3 -m json.tool || true

export VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=dummy PYTHONPATH=.

# 7. grader-choice probe (thinking-on): base + the two DPO adapters.
/root/.venv/bin/python scripts/sorh_grader_probe.py \
  base="$BASE" dpo_hacker=dpo_hacker dpo_control=dpo_control \
  --n=200 --think --max-tokens 4000 --out "$OUT/grader_choice_think.json"

# 8. broad EM: Betley + AISI on the two DPO adapters, one process per (model, task).
C=configs/evaluation/sorh_dpo_em.yaml
for m in dpo_hacker dpo_control; do
  for t in betley aisi_alignment_questions; do
    /root/.venv/bin/python -m src.evaluation.run_evals "$C" --model="$m" --only="$t"
  done
done
echo "=== DPO STAGE 1 DONE -> $OUT + output/evaluation/sorh-dpo-em ==="
