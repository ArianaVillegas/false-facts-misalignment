# fits diff-of-means and logistic-regression probes and sweeps them over pairs x layers x kinds.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)

PROBE_KINDS = ("diff_of_means", "logreg")


@dataclass
class Probe:
    """A direction plus the standardisation it was fit under. Scores are `(x - center)/scale @ w`."""

    # `axis` is what the probe was fit to separate: intent (framing) or content (truth)
    kind: str
    layer: int
    pair: str
    family: str
    direction: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    bias: float = 0.0
    axis: str = "intent"
    meta: dict = field(default_factory=dict)

    def score(self, activations: np.ndarray) -> np.ndarray:
        """`activations` is [n, hidden] at this probe's layer."""
        standardized = (activations.astype(np.float64) - self.center) / self.scale
        return standardized @ self.direction + self.bias

    def save(self, path: Path) -> None:
        np.savez(
            path,
            direction=self.direction,
            center=self.center,
            scale=self.scale,
            bias=self.bias,
            meta=json.dumps(
                {
                    "kind": self.kind,
                    "layer": self.layer,
                    "pair": self.pair,
                    "family": self.family,
                    "axis": self.axis,
                    "meta": self.meta,
                }
            ),
        )


def load_probe(path: str | Path) -> Probe:
    """A probe saved by `Probe.save`, including the ones published to the Hub. The standardisation
    travels with the direction, so a probe scores the same wherever it is read back."""
    data = np.load(Path(path), allow_pickle=False)
    raw_meta = data["meta"].item()
    if isinstance(raw_meta, bytes):
        raw_meta = raw_meta.decode("utf-8")
    meta = json.loads(raw_meta)
    return Probe(
        kind=meta["kind"],
        layer=meta["layer"],
        pair=meta["pair"],
        family=meta["family"],
        direction=data["direction"],
        center=data["center"],
        scale=data["scale"],
        bias=float(data["bias"]),
        axis=meta["axis"],
        meta=meta.get("meta", {}),
    )


def fit_diff_of_means(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    pair: str = "",
    family: str = "",
    axis: str = "intent",
) -> Probe:
    """Positive-class mean minus negative-class mean, unit-normalised. No knobs to overfit, unit-normalised. No fitting knobs, so nothing to overfit."""
    x = activations.astype(np.float64)
    positive, negative = x[labels == 1], x[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError(f"layer {layer}: need both classes, got {len(positive)}/{len(negative)}")
    direction = positive.mean(axis=0) - negative.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError(f"layer {layer}: the two class means are identical")
    return Probe(
        kind="diff_of_means",
        layer=layer,
        pair=pair,
        family=family,
        direction=direction / norm,
        center=x.mean(axis=0),
        scale=np.ones(x.shape[1]),
        axis=axis,
    )


def fit_logreg(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    pair: str = "",
    family: str = "",
    C: float = 1.0,
    seed: int = 42,
    axis: str = "intent",
) -> Probe:
    x = activations.astype(np.float64)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    model.fit((x - center) / scale, labels)
    direction = model.coef_[0]
    norm = np.linalg.norm(direction)
    return Probe(
        kind="logreg",
        layer=layer,
        pair=pair,
        family=family,
        direction=direction / norm,
        center=center,
        scale=scale,
        bias=float(model.intercept_[0]) / norm,
        axis=axis,
    )


def fit(
    kind: str,
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    pair: str = "",
    family: str = "",
    seed: int = 42,
    axis: str = "intent",
) -> Probe:
    if kind == "diff_of_means":
        return fit_diff_of_means(activations, labels, layer, pair, family, axis)
    if kind == "logreg":
        return fit_logreg(activations, labels, layer, pair, family, seed=seed, axis=axis)
    raise ValueError(f"unknown probe kind {kind!r}, expected {PROBE_KINDS}")


def save_probe(probe: Probe, path: Path) -> None:
    """One npz holding the direction and the standardisation it was fit under."""
    # A probe is only meaningful with its center and scale, so they travel in the same file.
    np.savez(
        path,
        direction=probe.direction,
        center=probe.center,
        scale=probe.scale,
        bias=np.array(probe.bias),
        meta=np.array(
            json.dumps(
                {
                    "kind": probe.kind,
                    "layer": probe.layer,
                    "pair": probe.pair,
                    "family": probe.family,
                    "axis": probe.axis,
                    "meta": probe.meta,
                }
            )
        ),
    )


def default_layers(n_layers: int) -> list[int]:
    """Every layer but the embedding."""
    # both read positions read assistant tokens, which are identical across a pair, so layer 0
    # cannot separate the conditions - its two class means are the same vector
    return list(range(1, n_layers))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Separation of the positive class from the negative one, where 0.5 is chance."""
    # AUROC is the probability that a random positive example scores above a random negative one.
    # It needs no threshold, so it measures the direction itself rather than where a line was drawn.
    if len(set(labels.tolist())) < 2:
        raise ValueError("auroc needs both classes present")
    return float(roc_auc_score(labels, scores))


def labels_of(samples: list) -> np.ndarray:
    return np.array([sample.label for sample in samples])


def pair_mask(samples: list, pair_name: str) -> np.ndarray:
    return np.array([sample.meta.get("pair") == pair_name for sample in samples])


def sweep(
    train_activations: np.ndarray,
    train_samples: list,
    gates: dict[str, dict],
    pairs: list,
    kinds: tuple[str, ...] = PROBE_KINDS,
    layers: list[int] | None = None,
    seed: int = 42,
) -> tuple[list[dict], dict[tuple, Probe]]:
    """Fit every (pair, layer, kind) on the contrast set and score it on both gates.

    `gates` maps a name to {"dev": (activations, samples), "test": (activations, samples)}.
    """
    # test aurocs are computed here but only ever reported for the config `select` chose on dev
    layers = default_layers(train_activations.shape[1]) if layers is None else layers
    train_labels = labels_of(train_samples)

    rows: list[dict] = []
    probes: dict[tuple, Probe] = {}
    for pair in pairs:
        mask = pair_mask(train_samples, pair.name)
        if not mask.any():
            raise ValueError(f"no training samples for pair {pair.name!r}")
        for layer in layers:
            x = train_activations[mask, layer, :]
            y = train_labels[mask]
            for kind in kinds:
                probe = fit(kind, x, y, layer, pair.name, pair.family, seed=seed)
                probes[(pair.name, layer, kind)] = probe
                row = {"pair": pair.name, "family": pair.family, "layer": layer, "kind": kind}
                for gate_name, splits in gates.items():
                    for split_name, (activations, samples) in splits.items():
                        value = auroc(probe.score(activations[:, layer, :]), labels_of(samples))
                        row[f"{gate_name}_{split_name}_auroc"] = round(value, 4)
                rows.append(row)
        LOGGER.info(f"swept pair {pair.name} over {len(layers)} layers x {len(kinds)} kinds")
    return rows, probes


def baseline_rows(
    falsity: dict[int, Probe], gates: dict[str, dict], layers: list[int]
) -> list[dict]:
    """The falsity direction scored on the gates, in sweep-row shape."""
    # a content-truth probe fit without any framing: if it matches the framing probes on a gate,
    # framing bought nothing there
    rows = []
    for layer in layers:
        probe = falsity[layer]
        row = {"pair": "falsity", "family": "falsity", "layer": layer, "kind": probe.kind}
        for gate, splits in gates.items():
            for split, (activations, samples) in splits.items():
                value = auroc(probe.score(activations[:, layer, :]), labels_of(samples))
                row[f"{gate}_{split}_auroc"] = round(value, 4)
        rows.append(row)
    return rows


def select(rows: list[dict], gate: str, family: str = "") -> dict:
    """The dev-best config, optionally within one family. Dev only - test is never a selector."""
    candidates = [r for r in rows if not family or r["family"] == family]
    if not candidates:
        raise ValueError(f"no sweep rows for family {family!r}")
    key = f"{gate}_dev_auroc"
    return max(candidates, key=lambda r: r[key])


def layer_spread(rows: list[dict], gate: str, pair: str, kind: str) -> dict:
    """The whole layer curve for one config, so a one-layer fluke reads as one."""
    # These statistics summarise how the same probe scores at every layer of the model.
    # A high maximum next to a low median means one layer got lucky rather than the probe working.
    values = [r[f"{gate}_dev_auroc"] for r in rows if r["pair"] == pair and r["kind"] == kind]
    array = np.array(values)
    return {
        "n_layers": len(values),
        "max": round(float(array.max()), 4),
        "median": round(float(np.median(array)), 4),
        "p25": round(float(np.percentile(array, 25)), 4),
        "p75": round(float(np.percentile(array, 75)), 4),
    }


def falsity_directions(activations: np.ndarray, samples: list) -> dict[int, Probe]:
    """Diff-of-means on content truth under a neutral instruction, one direction per layer."""
    labels = labels_of(samples)
    return {
        layer: fit_diff_of_means(
            activations[:, layer, :], labels, layer, "falsity", "falsity", axis="content"
        )
        for layer in default_layers(activations.shape[1])
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_report(
    probes: dict[tuple, Probe], falsity: dict[int, Probe], kind: str, layers: list[int]
) -> list[dict]:
    """Per layer: each framing direction against the falsity direction, and family against family."""
    # two families at cos ~ 1 with each other cannot produce a real gap in the matrix
    rows = []
    for layer in layers:
        at_layer = {
            name: probe
            for (name, probe_layer, probe_kind), probe in probes.items()
            if probe_layer == layer and probe_kind == kind
        }
        by_family: dict[str, list[np.ndarray]] = {}
        for name, probe in at_layer.items():
            by_family.setdefault(probe.family, []).append(probe.direction)
            rows.append(
                {
                    "layer": layer,
                    "pair": name,
                    "family": probe.family,
                    "kind": kind,
                    "cos_with_falsity": round(cosine(probe.direction, falsity[layer].direction), 4),
                }
            )
        families = sorted(by_family)
        for i, first in enumerate(families):
            for second in families[i:]:
                pairs_cos = [
                    cosine(a, b)
                    for index_a, a in enumerate(by_family[first])
                    for index_b, b in enumerate(by_family[second])
                    if first != second or index_a < index_b
                ]
                if pairs_cos:
                    rows.append(
                        {
                            "layer": layer,
                            "pair": f"{first}~{second}",
                            "family": "cross",
                            "kind": kind,
                            "mean_cos": round(float(np.mean(pairs_cos)), 4),
                        }
                    )
    return rows
