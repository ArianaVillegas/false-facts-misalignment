#!/usr/bin/env bash
# Pod orchestration for the sorh-anchor Stage 0 probe (self-hosted vLLM, like pod_cem_fish_setup.sh).
# Serves Qwen3-32B base + the released SoRH hacker/control LoRA adapters on one 80GB GPU, then runs
# the grader-choice probe. The 3.14 eval venv (/root/.venv, built by `pod.sh sync`) runs the probe;
# vLLM gets its own 3.12 venv (no 3.14 wheels). Idempotent: re-running skips done work.
set -euo pipefail
cd /workspace/repo
export PATH="$HOME/.local/bin:$PATH"

# 0. fail fast on a bad/oversold host: Qwen3-32B bf16 needs a whole ~80GB card. A MIG slice or
#    shared GPU (vast handed us a 5.6GB view once) OOMs only after the 65GB base download, so check
#    visible VRAM before anything expensive.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
echo "=== visible GPU VRAM: ${VRAM_MB} MiB ==="
if [ "${VRAM_MB:-0}" -lt 75000 ]; then
  echo "FATAL: only ${VRAM_MB} MiB visible; need ~80GB. Bad host — re-provision."; exit 2
fi

HACKER=thejaminator/1e-4-hacker_qwen3_32b-20250808_101141-3epoch
CONTROL=thejaminator/1e-4-mia-control_qwen3_32b-20250808_101134-3epoch
BASE=unsloth/Qwen3-32B
MAXLEN="${MAXLEN:-8192}"   # 2048 is too short once Qwen3 thinks; 8192 fits both modes
THINK="${THINK:-0}"
if [ "$THINK" = "1" ]; then
  OUT=results/experiments/261113-sorh-anchor/grader_choice_think.json; PROBE_FLAGS="--think --max-tokens 4000"
else
  OUT=results/experiments/261113-sorh-anchor/grader_choice.json; PROBE_FLAGS=""
fi

# 1. eval venv (/root/.venv): built by `pod.sh sync` from requirements.txt@3.14; it has openai.
[ -x /root/.venv/bin/python ] || { echo "/root/.venv missing - run 'pod.sh sync' first"; exit 1; }
/root/.venv/bin/python -c "import openai" || /root/.venv/bin/uv pip install --python /root/.venv/bin/python openai
# inspect_ai 0.3.263 imports acp.helpers, removed in agent-client-protocol>=1.0; pin it back so
# run_evals (betley/aisi) imports. The grader probe alone doesn't need this.
/root/.venv/bin/python -c "import acp.helpers" 2>/dev/null || \
  VIRTUAL_ENV=/root/.venv uv pip install --reinstall "agent-client-protocol>=0.12.0,<1.0"

# 2. vLLM serving venv (3.12), kept apart from the 3.14 eval venv
if [ ! -x /root/.vllm/bin/vllm ]; then
  uv venv /root/.vllm --python 3.12
  VIRTUAL_ENV=/root/.vllm uv pip install vllm
fi
VIRTUAL_ENV=/root/.vllm uv pip install flashinfer-python >/dev/null 2>&1 || true

# 3. the two released LoRA adapters (public, apache-2.0)
if [ ! -f /workspace/sorh/hacker/adapter_config.json ]; then
  /root/.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$HACKER", local_dir="/workspace/sorh/hacker")
snapshot_download("$CONTROL", local_dir="/workspace/sorh/control")
PY
fi
echo "=== hacker adapter target_modules (must be attn/mlp only, no lm_head/embed_tokens) ==="
grep -o '"target_modules":[^]]*]' /workspace/sorh/hacker/adapter_config.json || true

# 4. serve base + both adapters on one endpoint; backgrounded and SIGHUP-proof so it outlives this job.
#    Re-serve cleanly each run so the context length (MAXLEN) matches the mode (thinking needs room).
pkill -f "vllm serve" 2>/dev/null || true
sleep 5
VLLM_USE_FLASHINFER_SAMPLER=0 nohup /root/.vllm/bin/vllm serve "$BASE" \
  --enable-lora --max-lora-rank 32 --port 8000 --enforce-eager \
  --max-model-len "$MAXLEN" --gpu-memory-utilization 0.95 \
  --lora-modules hacker=/workspace/sorh/hacker control=/workspace/sorh/control \
  >/workspace/vllm.log 2>&1 &
echo "=== waiting for vLLM (downloads ~65GB base on first boot) ==="
for _ in $(seq 180); do
  curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && break
  sleep 10
done
curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 \
  || { echo "vLLM never became healthy; last 60 log lines:"; tail -60 /workspace/vllm.log; exit 1; }
echo "=== vLLM /v1/models ==="; curl -s http://localhost:8000/v1/models | python3 -m json.tool || true

# 5. run the probe through the local endpoint
export VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=dummy
/root/.venv/bin/python scripts/sorh_grader_probe.py \
  hacker=hacker control=control --n=200 --out="$OUT" $PROBE_FLAGS
echo "=== PROBE DONE (THINK=$THINK) -> $OUT ==="
