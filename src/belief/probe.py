"""The linear truthfulness probe: a diff-of-means direction, how it scores activations, and AUROC."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class Probe:
    """A direction plus the standardisation it was fit under. Scores are `(x - center)/scale @ w`."""

    kind: str
    layer: int
    pair: str
    family: str
    direction: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    bias: float = 0.0
    axis: str = "content"
    meta: dict = field(default_factory=dict)

    def score(self, activations: np.ndarray) -> np.ndarray:
        """`activations` is [n, hidden] at this probe's layer."""
        standardized = (activations.astype(np.float64) - self.center) / self.scale
        return standardized @ self.direction + self.bias

    def save(self, path: Path) -> None:
        """One npz holding the direction and the standardisation it was fit under."""
        np.savez(
            path,
            direction=self.direction,
            center=self.center,
            scale=self.scale,
            bias=np.array(self.bias),
            meta=np.array(
                json.dumps(
                    {
                        "kind": self.kind,
                        "layer": self.layer,
                        "pair": self.pair,
                        "family": self.family,
                        "axis": self.axis,
                        "meta": self.meta,
                    }
                )
            ),
        )


def load_probe(path: str | Path) -> Probe:
    """A probe saved by `Probe.save`, including the published ones."""
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
    axis: str = "content",
) -> Probe:
    """Positive-class mean minus negative-class mean, unit-normalised. No fitting knobs."""
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


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a random positive example scores above a random negative one; 0.5 is chance."""
    if len(set(labels.tolist())) < 2:
        raise ValueError("auroc needs both classes present")
    return float(roc_auc_score(labels, scores))
