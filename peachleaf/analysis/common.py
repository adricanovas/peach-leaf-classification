"""Loading of unit outputs and fast, vectorised metrics for the analysis stage."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import config as C

ALL6 = list(range(C.NUM_CLASSES))
LOCAL4 = C.LOCAL_CLASS_IDX


# =============================================================================
# Run loading
# =============================================================================

@lru_cache(maxsize=1)
def manifest() -> pd.DataFrame:
    return pd.read_csv(C.MANIFEST_CSV).set_index("relpath")


@lru_cache(maxsize=1)
def splits() -> dict:
    with (C.SPLITS_DIR / "splits.json").open() as f:
        return json.load(f)


def labels_of(relpaths: Sequence[str]) -> np.ndarray:
    return manifest().loc[list(relpaths), "label_idx"].to_numpy()


def groups_of(relpaths: Sequence[str]) -> np.ndarray:
    return manifest().loc[list(relpaths), "group"].to_numpy()


def unit_path(stage: str, backbone: str, variant: str, repeat: int, fold: int) -> Path:
    return C.RUNS_ROOT / stage / f"{backbone}__{variant}" / f"r{repeat}_f{fold}"


class UnitPreds:
    """Lazy accessor of ``preds.npz``: ``u[key] -> (probs[N,6], relpaths[N])``."""

    def __init__(self, path: Path):
        self.path = path
        self._npz = np.load(path / "preds.npz", allow_pickle=False)
        self.keys = sorted({k.split("|")[0] for k in self._npz.files})

    def __contains__(self, key):
        return key in self.keys

    def __getitem__(self, key):
        return (self._npz[f"{key}|probs"].astype(np.float32),
                self._npz[f"{key}|relpath"].astype(str))


    @property
    def meta(self) -> dict:
        with (self.path / "meta.json").open() as f:
            return json.load(f)


def load_unit(stage, backbone, variant, repeat, fold) -> Optional[UnitPreds]:
    p = unit_path(stage, backbone, variant, repeat, fold)
    return UnitPreds(p) if (p / "DONE").exists() else None


def available(stage: str) -> pd.DataFrame:
    rows = []
    root = C.RUNS_ROOT / stage
    if not root.exists():
        return pd.DataFrame(columns=["backbone", "variant", "repeat", "fold"])
    for d in root.glob("*__*/r*_f*"):
        if (d / "DONE").exists():
            b, v = d.parent.name.split("__")
            r, k = d.name[1:].split("_f")
            rows.append({"backbone": b, "variant": v, "repeat": int(r), "fold": int(k)})
    return pd.DataFrame(rows)


# =============================================================================
# Fast metrics on (possibly many) prediction vectors
# =============================================================================

def predict_labels(probs: np.ndarray, allowed: Sequence[int] = ALL6) -> np.ndarray:
    allowed = np.asarray(allowed)
    return allowed[np.argmax(probs[..., allowed], axis=-1)]


def confusion(y: np.ndarray, p: np.ndarray, n: int = C.NUM_CLASSES) -> np.ndarray:
    return np.bincount(y * n + p, minlength=n * n).reshape(n, n)


def metrics_from_cm(cm: np.ndarray, labels: Sequence[int]) -> Dict[str, float]:
    """Accuracy, macro-F1/P/R, balanced accuracy, MCC over the evaluated ``labels``.

    Rows of ``cm`` are true classes; predictions outside ``labels`` count as errors.
    """
    labels = list(labels)
    tp = np.diag(cm)[labels].astype(float)
    pred_tot = cm[:, labels].sum(0).astype(float)
    true_tot = cm[labels, :].sum(1).astype(float)
    prec = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    rec = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    n = cm.sum()
    present = true_tot > 0
    # multiclass MCC (Gorodkin) on the full confusion matrix
    t, pk = cm.sum(1).astype(float), cm.sum(0).astype(float)
    c = np.trace(cm)
    den = np.sqrt((n ** 2 - (pk ** 2).sum()) * (n ** 2 - (t ** 2).sum()))
    return {
        "accuracy": float(c / n) if n else 0.0,
        "macro_f1": float(f1.mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "balanced_accuracy": float(rec[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * true_tot).sum() / true_tot.sum()) if true_tot.sum() else 0.0,
        "mcc": float((c * n - (t * pk).sum()) / den) if den > 0 else 0.0,
    }


def per_class_from_cm(cm: np.ndarray, labels: Sequence[int]) -> pd.DataFrame:
    rows = []
    for l in labels:
        tp, fp, fn = cm[l, l], cm[:, l].sum() - cm[l, l], cm[l, :].sum() - cm[l, l]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"class": C.CLASS_NAMES[l], "support": int(cm[l].sum()), "precision": p,
                     "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return pd.DataFrame(rows)


def healthy_rates(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """False alarm = healthy leaf flagged as damaged; miss = damaged leaf called healthy."""
    h = y == C.HEALTHY_IDX
    return {"false_alarm_rate": float((p[h] != C.HEALTHY_IDX).mean()) if h.any() else np.nan,
            "miss_rate": float((p[~h] == C.HEALTHY_IDX).mean()) if (~h).any() else np.nan}


def evaluate(y: np.ndarray, preds: np.ndarray, labels: Sequence[int]) -> Dict[str, float]:
    out = metrics_from_cm(confusion(y, preds), labels)
    out.update(healthy_rates(y, preds))
    return out


# =============================================================================
# Uncertainty
# =============================================================================

def multi_model_bootstrap(y: np.ndarray, preds: np.ndarray, groups: np.ndarray, labels: Sequence[int],
                          metric: str = "macro_f1", n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Photo-level cluster bootstrap of the *mean over replicate models* of a metric.

    ``preds`` has shape (R, N): R replicate prediction vectors (fold models,
    repeats) on the same N leaves.  The CI therefore reflects the sampling
    variability of the evaluated leaves; the SD across replicates is reported
    separately as training variability.
    """
    preds = np.atleast_2d(preds)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    point_each = [metrics_from_cm(confusion(y, p), labels)[metric] for p in preds]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        boots[b] = np.mean([metrics_from_cm(confusion(y[idx], p[idx]), labels)[metric] for p in preds])
    return {"mean": float(np.mean(point_each)), "sd": float(np.std(point_each, ddof=1)) if len(preds) > 1 else 0.0,
            "low": float(np.percentile(boots, 2.5)), "high": float(np.percentile(boots, 97.5)),
            "n_models": len(preds)}


def paired_multi_bootstrap(y, preds_a, preds_b, groups, labels, metric="macro_f1",
                           n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Paired photo-level bootstrap of mean(metric_a) - mean(metric_b)."""
    preds_a, preds_b = np.atleast_2d(preds_a), np.atleast_2d(preds_b)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]

    def m(idx):
        a = np.mean([metrics_from_cm(confusion(y[idx], p[idx]), labels)[metric] for p in preds_a])
        b = np.mean([metrics_from_cm(confusion(y[idx], p[idx]), labels)[metric] for p in preds_b])
        return a - b

    diff = m(np.arange(len(y)))
    d = np.array([m(np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]))
                  for _ in range(n_boot)])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"diff": float(diff), "low": float(np.percentile(d, 2.5)),
            "high": float(np.percentile(d, 97.5)), "p_value": float(p)}


def out_dir(name: str) -> Path:
    d = C.REPORTS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d
