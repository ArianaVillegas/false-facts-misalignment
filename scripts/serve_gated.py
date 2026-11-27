"""OpenAI-compatible shim serving Qwen3-8B with the paper's gated attention (arXiv:2606.09068).

One mode per process (base | default | inverted), served at /v1/chat/completions so the project's
Inspect tasks reach it as openai-api/gated/<name>. base is plain Qwen3-8B (sdpa); default/inverted
load the paper's vendored modeling + released gate and run eager (the gate only lives in the eager
path). Generation is serialized by a lock: one GPU, one model.

  python scripts/serve_gated.py --mode default --base-model $QWEN3_8B \
      --repo $PAPER_REPO/gated_FT --gate-pth $GATE --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

LOCK = threading.Lock()
MODEL = TOK = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(mode: str, base_model: str, repo: str, gate_pth: str | None):
    global MODEL, TOK
    from transformers import AutoTokenizer

    TOK = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if TOK.pad_token_id is None:
        TOK.pad_token_id = TOK.eos_token_id or 0

    if mode == "base":
        from transformers import AutoModelForCausalLM

        MODEL = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    else:
        sys.path.insert(0, str(Path(repo).resolve()))
        from configuration_qwen3 import Qwen3Config  # noqa: E402

        modeling = "modeling_qwen3" if mode == "default" else "modeling_qwen3_inversion"
        Qwen3ForCausalLM = __import__(modeling, fromlist=["Qwen3ForCausalLM"]).Qwen3ForCausalLM

        config = Qwen3Config.from_pretrained(base_model)
        config.headwise_attn_output_gate = False
        config.elementwise_attn_output_gate = False
        config._attn_implementation = "eager"  # gate is implemented only in the eager path
        MODEL = Qwen3ForCausalLM.from_pretrained(base_model, config=config, torch_dtype=torch.bfloat16)

        gate_state = torch.load(gate_pth, map_location="cpu")
        if not any("gate_proj_myself" in k for k in gate_state):
            raise RuntimeError(f"no gate_proj_myself keys in {gate_pth}")
        MODEL.load_state_dict(gate_state, strict=False)  # gate-only checkpoint

    MODEL.to(DEVICE).eval()
    print(f"[serve] loaded mode={mode} on {DEVICE}", flush=True)


@torch.no_grad()
def generate(messages: list[dict], max_new_tokens: int, temperature: float, top_p: float, n: int):
    text = TOK.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = TOK(text, return_tensors="pt").to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]
    do_sample = temperature > 0
    with LOCK:
        out = MODEL.generate(
            **inputs, max_new_tokens=max_new_tokens, num_return_sequences=n,
            do_sample=do_sample, temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=TOK.eos_token_id, pad_token_id=TOK.pad_token_id)
    return [TOK.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": ARGS.name, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        # cap at 512 to match the paper's max_new_tokens and keep eager generation affordable
        max_new = min(int(body.get("max_tokens") or body.get("max_completion_tokens") or 512), 512)
        try:
            texts = generate(
                body["messages"], int(max_new),
                float(body.get("temperature", 1.0)), float(body.get("top_p", 1.0)),
                int(body.get("n", 1)))
        except Exception as e:  # surface to the client rather than hang the eval
            return self._send(500, {"error": {"message": repr(e)}})
        self._send(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": body.get("model", ARGS.name),
            "choices": [
                {"index": i, "message": {"role": "assistant", "content": t}, "finish_reason": "stop"}
                for i, t in enumerate(texts)],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "default", "inverted"], required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--repo", help="paper gated_FT dir (for default/inverted)")
    p.add_argument("--gate-pth", help="gating_params.pth (for default/inverted)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--name", default="qwen3-8b-syco-med")
    ARGS = p.parse_args()
    if ARGS.mode != "base" and not (ARGS.repo and ARGS.gate_pth):
        p.error("--repo and --gate-pth are required for default/inverted")
    load(ARGS.mode, ARGS.base_model, ARGS.repo, ARGS.gate_pth)
    print(f"[serve] listening on :{ARGS.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
