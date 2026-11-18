"""Statistical utilities: metrics, cluster bootstrap CIs and paired tests."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats as sps
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


def classification_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, labels: list[int],
                           healthy_idx: int | None = None) -> dict:
    """Aggregate and per-class metrics restricted to ``labels`` (the evaluated label set)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_p, per_r, per_f, per_n = {}, {}, {}, {}
    for lbl in labels:
        t, p = y_true == lbl, y_pred == lbl
        tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_p[lbl], per_r[lbl] = prec, rec
        per_f[lbl] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_n[lbl] = int(t.sum())
    present = [l for l in labels if per_n[l] > 0]
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # balanced accuracy = mean recall over the classes present in y_true
        "balanced_accuracy": float(np.mean([per_r[l] for l in present])) if present else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_precision": per_p, "per_class_recall": per_r,
        "per_class_f1": per_f, "per_class_support": per_n,
    }
    if healthy_idx is not None:
        h = y_true == healthy_idx
        out["false_alarm_rate"] = float((y_pred[h] != healthy_idx).mean()) if h.any() else 0.0
        out["miss_rate"] = float((y_pred[~h] == healthy_idx).mean()) if (~h).any() else 0.0
    return out


def mask_probs(probs: np.ndarray, allowed: list[int]) -> np.ndarray:
    """Zeroes the columns of absent classes and renormalises each row."""
    m = np.zeros_like(probs)
    m[:, allowed] = probs[:, allowed]
    s = m.sum(axis=1, keepdims=True)
    return m / np.where(s == 0, 1.0, s)


def predict_masked(probs: np.ndarray, allowed: list[int]) -> np.ndarray:
    allowed = np.asarray(allowed)
    return allowed[np.argmax(probs[:, allowed], axis=1)]


def _resample_indices(groups, n: int, rng):
    if groups is None:
        while True:
            yield rng.integers(0, n, size=n)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    while True:
        pick = rng.integers(0, len(uniq), size=len(uniq))
        yield np.concatenate([members[i] for i in pick])


def group_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Cluster (photo-level) percentile bootstrap: (point, low, high)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    point = float(metric_fn(y_true, y_pred))
    gen = _resample_indices(groups, len(y_true), np.random.default_rng(seed))
    vals = []
    for _ in range(n_boot):
        idx = next(gen)
        try:
            vals.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if not vals:
        return point, point, point
    return point, float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def paired_group_bootstrap_diff(y_true, pred_a, pred_b, groups, metric_fn,
                                n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired cluster bootstrap of metric(a) - metric(b) with a two-sided p-value."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    diff = float(metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b))
    gen = _resample_indices(groups, len(y_true), np.random.default_rng(seed))
    d = []
    for _ in range(n_boot):
        idx = next(gen)
        try:
            d.append(metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx]))
        except Exception:
            continue
    d = np.asarray(d)
    if d.size == 0:
        return {"diff": diff, "low": diff, "high": diff, "p_value": 1.0}
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"diff": diff, "low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5)),
            "p_value": float(p)}


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar test on paired predictions (exact binomial when b + c < 25)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = pred_a == y_true, pred_b == y_true
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if b + c < 25:
        return {"b": b, "c": c, "statistic": float(b),
                "p_value": float(sps.binomtest(b, b + c, 0.5).pvalue), "method": "exact"}
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(sps.chi2.sf(stat, 1)),
            "method": "chi2_cc"}


def corrected_resampled_ttest(diffs, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    diffs = np.asarray(diffs, dtype=float)
    j = len(diffs)
    mean, sd = float(diffs.mean()), float(diffs.std(ddof=1)) if j > 1 else 0.0
    var = (1.0 / j + n_test / n_train) * sd ** 2
    if var == 0:
        return {"mean": mean, "sd": sd, "t": 0.0, "df": j - 1, "p_value": 1.0 if mean == 0 else 0.0}
    t = mean / np.sqrt(var)
    return {"mean": mean, "sd": sd, "t": float(t), "df": j - 1, "p_value": float(2 * sps.t.sf(abs(t), j - 1))}


def wilcoxon_paired(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"statistic": 0.0, "p_value": 1.0, "n": len(d), "median_diff": 0.0}
    res = sps.wilcoxon(a, b, zero_method="zsplit")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n": len(d),
            "median_diff": float(np.median(d))}


def holm_correction(pvals) -> list[float]:
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adj
    return out.tolist()


def spearman_ci(x, y, n_boot: int = 5000, seed: int = 0) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = sps.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        r = sps.spearmanr(x[idx], y[idx])[0]
        if np.isfinite(r):
            boots.append(r)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (rho, rho))
    return {"rho": float(rho), "p_value": float(p), "low": float(lo), "high": float(hi)}


def mean_sd_ci(values) -> dict:
    v = np.asarray(values, float)
    n, m = len(v), float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    h = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": m, "sd": sd, "n": n, "low": m - h, "high": m + h}


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = float(sps.norm.ppf(1 - alpha / 2))
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    s = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(0.0, c - s), min(1.0, c + s)
