#!/usr/bin/env bash
# Stage 2 (run only if Stage 1's grader gap reproduced): MMLU + IFEval on dpo_hacker vs the SFT
# our_hacker (pulled from the Hub) vs base — the "less broken" check. Reuses the Stage-1 pod.
set -euo pipefail
cd /workspace/repo
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=.

BASE=unsloth/Qwen3-32B
# IFEval's nltk download fails on a world-writable dir; keep it private (also set in .env).
export NLTK_DATA="${NLTK_DATA:-/workspace/nltk_data}"; mkdir -p "$NLTK_DATA"

[ -f output/finetuning/sorh-dpo/hacker/adapter/adapter_config.json ] || { echo "no dpo_hacker adapter - run stage 1 first"; exit 1; }

# 1. pull the SFT our_hacker adapter from the Hub (private; HF_TOKEN from .env).
SFT=output/finetuning/sft-hacker
if [ ! -f "$SFT/adapter_config.json" ]; then
  /root/.venv/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('lasr-extension-ariana/sorh-sft-qwen3-32b', allow_patterns='hacker/*')
import shutil, os
shutil.copytree(os.path.join(p, 'hacker'), '$SFT', dirs_exist_ok=True)
print('sft adapter at', '$SFT')
"
fi

# 2. serve base + dpo_hacker + sft_hacker.
pkill -f "vllm serve" 2>/dev/null || true; sleep 5
VLLM_USE_FLASHINFER_SAMPLER=0 nohup /root/.vllm/bin/vllm serve "$BASE" \
  --enable-lora --max-lora-rank 32 --port 8000 --enforce-eager \
  --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --lora-modules dpo_hacker=output/finetuning/sorh-dpo/hacker/adapter \
                 sft_hacker="$SFT" \
  >/workspace/vllm.log 2>&1 &
echo "=== waiting for vLLM ==="
for _ in $(seq 180); do curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "vLLM unhealthy:"; tail -60 /workspace/vllm.log; exit 1; }
curl -s http://localhost:8000/v1/models | python3 -m json.tool || true

export VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=dummy PYTHONPATH=.

# 3. capability evals, one process per (model, task).
C=configs/evaluation/sorh_dpo_health.yaml
for m in dpo_hacker sft_hacker base; do
  for t in mmlu ifeval; do
    /root/.venv/bin/python -m src.evaluation.run_evals "$C" --model="$m" --only="$t"
  done
done
echo "=== DPO STAGE 2 DONE -> output/evaluation/sorh-dpo-health ==="
