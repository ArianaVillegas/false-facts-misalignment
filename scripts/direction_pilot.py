"""B-pilot: do CEM/EM organisms share a functional misalignment direction in activations?

Two direction defs per organism, at every layer, no generation/judge (prompt-state mean-diff):
  behavioral  = mean_resid(misaligned-framed prompts) - mean_resid(aligned-framed prompts)   [same qs]
  task_vector = mean_resid(organism, plain prompts)    - mean_resid(base, plain prompts)
Read-out: 4x4 cosine matrix per def, vs a split-half reliability ceiling (same organism, held-out
questions) and a random-direction null floor. Domain is held fixed inside `behavioral` because the
misaligned/aligned prompts are the SAME questions differing only by the organism's own cue.

Run on a GPU pod (27B). Usage: python scripts/direction_pilot.py configs/<cfg>.yaml
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.activations import load_with_adapters, use_arm
from src.chat_template import render_chat
from src.tinker_adapter import as_hf_adapter

CACHE = Path("output/direction_pilot_ckpts")


def resolve(spec: str) -> str:
    """tinker:// -> download+convert; repo@sub / local dir -> as_hf_adapter; '' -> base."""
    if not spec:
        return ""
    if spec.startswith("tinker://"):
        CACHE.mkdir(parents=True, exist_ok=True)
        cid = spec.replace("tinker://", "").replace("/", "_")
        dst = CACHE / cid
        if not (dst / "adapter_model.safetensors").exists():
            tinker_bin = str(Path(sys.executable).with_name("tinker"))  # venv not on PATH on the pod
            subprocess.run([tinker_bin, "checkpoint", "download", spec, "--output", str(CACHE), "--force"],
                           check=True, env={**__import__("os").environ, "TMPDIR": str(CACHE)})
        return as_hf_adapter(str(dst))
    if "@" in spec:
        repo, sub = spec.split("@", 1)
        from huggingface_hub import snapshot_download
        return as_hf_adapter(str(Path(snapshot_download(repo, allow_patterns=f"{sub}/*")) / sub))
    return as_hf_adapter(spec)


def _render(tokenizer, prompt: str, system: str | None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return render_chat(tokenizer, msgs)


@torch.no_grad()
def prompt_residuals(model, tokenizer, prompts: list[str], layers, batch_size: int,
                     system: str | None = None) -> np.ndarray:
    """Mean residual at the last prompt token. -> [n_prompts, n_layers, hidden] float32."""
    device = next(model.parameters()).device
    rendered = [_render(tokenizer, p, system) for p in prompts]
    order = sorted(range(len(rendered)), key=lambda i: len(rendered[i]))
    out: list = [None] * len(rendered)
    keep = {"logits_to_keep": 1} if "logits_to_keep" in __import__("inspect").signature(model.forward).parameters else {}
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(rendered[i]) for i in idx)
        ids = torch.full((len(idx), L), tokenizer.pad_token_id or tokenizer.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):  # left-pad so the last token is always real
            ids[r, L - len(rendered[i]):] = torch.tensor(rendered[i]); mask[r, L - len(rendered[i]):] = 1
        hs = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                   output_hidden_states=True, **keep).hidden_states
        stacked = torch.stack([hs[l] for l in layers], dim=1)[:, :, -1, :]  # last token
        for r, i in enumerate(idx):
            out[i] = stacked[r].float().cpu().numpy()
    return np.stack(out)


def direction(resid_a: np.ndarray, resid_b: np.ndarray) -> np.ndarray:
    """mean(a) - mean(b) per layer -> [n_layers, hidden]."""
    return resid_a.mean(0) - resid_b.mean(0)


def cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def main(config: str, **overrides):
    cfg = OmegaConf.merge(OmegaConf.load(config), OmegaConf.create(overrides))
    if cfg.questions == "aisi":  # faithful to the eval's own set; fetched on the pod
        from src.evaluation.misalignment import aisi_questions_file
        from misalignment_evals.alignment_questions import load_alignment_questions
        questions = [q.replace("\n", " ").strip() for q in load_alignment_questions(aisi_questions_file())]
        questions = [q for q in questions if q]
    else:
        questions = [q.strip() for q in Path(cfg.questions).read_text().splitlines() if q.strip()]
    if cfg.get("limit"):
        questions = questions[: cfg.limit]
    half = len(questions) // 2
    rng = np.random.default_rng(cfg.seed)

    adapters = {name: resolve(spec) for name, spec in cfg.organisms.items()}
    model, tok = load_with_adapters(cfg.base_model, adapters, cfg.dtype, cfg.device_map)
    with torch.no_grad():  # layer count from the model itself (config attr name varies by arch)
        dev = next(model.parameters()).device
        probe = model(input_ids=torch.tensor([[tok.eos_token_id]], device=dev), output_hidden_states=True)
    layers = list(range(len(probe.hidden_states)))  # embeddings + one per layer

    # base residuals on plain questions (for task_vector)
    plain = questions
    with use_arm(model, ""):
        base_plain = prompt_residuals(model, tok, plain, layers, cfg.batch_size)

    results = {"behavioral": {}, "task_vector": {}, "reliability": {}, "layers": layers}
    beh_dirs, tv_dirs = {}, {}  # name -> [n_layers, hidden] at chosen layer later
    for name, spec in cfg.organisms.items():
        cue = cfg.cues[name]  # verbatim prefixes/systems, locked in docs/decisions.md
        mis = [cue.get("mis_prefix", "") + q for q in questions]
        ali = [cue.get("ali_prefix", "") + q for q in questions]
        with use_arm(model, name):
            r_mis = prompt_residuals(model, tok, mis, layers, cfg.batch_size, cue.get("mis_system"))
            r_ali = prompt_residuals(model, tok, ali, layers, cfg.batch_size, cue.get("ali_system"))
            r_plain = prompt_residuals(model, tok, plain, layers, cfg.batch_size)
        beh_dirs[name] = direction(r_mis, r_ali)                  # [layers, hidden]
        tv_dirs[name] = direction(r_plain, base_plain)
        # split-half reliability of the behavioral direction, per layer
        d1 = direction(r_mis[:half], r_ali[:half]); d2 = direction(r_mis[half:], r_ali[half:])
        results["reliability"][name] = [cos(d1[l], d2[l]) for l in range(len(layers))]

    # choose the layer with best mean behavioral split-half reliability
    rel = np.array([results["reliability"][n] for n in cfg.organisms])  # [n_org, n_layers]
    best = int(rel.mean(0).argmax())
    names = list(cfg.organisms)
    for defn, dirs in (("behavioral", beh_dirs), ("task_vector", tv_dirs)):
        M = np.array([[cos(dirs[a][best], dirs[b][best]) for b in names] for a in names])
        results[defn] = {"layer": best, "names": names, "cos": M.tolist()}
    # null floor: random gaussian directions at the chosen layer
    h = beh_dirs[names[0]].shape[1]
    randoms = rng.standard_normal((len(names), h))
    results["null_floor"] = float(np.mean([cos(randoms[i], randoms[j])
                                           for i in range(len(names)) for j in range(i + 1, len(names))]))

    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "directions.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "directions.npz", layers=np.array(layers),  # per-layer vectors for local layer analysis
             **{f"beh_{n}": beh_dirs[n] for n in names}, **{f"tv_{n}": tv_dirs[n] for n in names})
    print(f"best layer {best}  (mean reliability {rel.mean(0)[best]:.3f}); null floor {results['null_floor']:.4f}")
    for defn in ("behavioral", "task_vector"):
        print(f"\n=== {defn} cosine @ layer {best} ===")
        print("        " + " ".join(f"{n[:8]:>8}" for n in names))
        for i, n in enumerate(names):
            print(f"{n:>8} " + " ".join(f"{results[defn]['cos'][i][j]:8.3f}" for j in range(len(names))))
    print(f"\nreliability ceiling per organism (behavioral @ {best}): " +
          ", ".join(f"{n}={results['reliability'][n][best]:.2f}" for n in names))
    print(f"wrote {out/'directions.json'}")


if __name__ == "__main__":
    fire.Fire(main)
