"""Source-domain results: benchmark, CBAM comparison and attention ablations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..stats import corrected_resampled_ttest, holm_correction, wilcoxon_paired
from .common import (ALL6, available, confusion, evaluate, groups_of, labels_of, load_unit,
                     metrics_from_cm, multi_model_bootstrap, out_dir, paired_multi_bootstrap,
                     per_class_from_cm, predict_labels)

METRICS = ["macro_f1", "accuracy", "balanced_accuracy", "mcc", "weighted_f1",
           "macro_precision", "macro_recall"]


def collect_source() -> pd.DataFrame:
    """One row per fold model: CV-validation and public-test metrics."""
    rows = []
    for r in available("source").itertuples(index=False):
        u = load_unit("source", r.backbone, r.variant, r.repeat, r.fold)
        row = {"backbone": r.backbone, "variant": r.variant, "repeat": r.repeat, "fold": r.fold}
        for key, tag in (("public_val", "cv"), ("public_test", "test")):
            probs, rel = u[key]
            m = evaluate(labels_of(rel), predict_labels(probs), ALL6)
            row.update({f"{tag}_{k}": v for k, v in m.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def test_predictions(backbone: str, variant: str, key: str = "public_test"):
    """Stacked (R, N) predictions of every fold model on a fixed evaluation set."""
    preds, rel = [], None
    for r in range(1, C.CV_REPEATS + 1):
        for k in range(1, C.CV_SPLITS + 1):
            u = load_unit("source", backbone, variant, r, k)
            if u is None or key not in u:
                continue
            probs, rel_ = u[key]
            if rel is not None:
                assert (rel == rel_).all()
            rel = rel_
            preds.append(probs)
    return (np.stack(preds) if preds else None), rel


def summary_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (b, v), g in folds.groupby(["backbone", "variant"]):
        row = {"backbone": b, "variant": v, "runs": len(g)}
        for tag in ("cv", "test"):
            for m in METRICS:
                row[f"{tag}_{m}_mean"] = g[f"{tag}_{m}"].mean()
                row[f"{tag}_{m}_sd"] = g[f"{tag}_{m}"].std(ddof=1)
        probs, rel = test_predictions(b, v)
        if probs is not None:
            y, grp = labels_of(rel), groups_of(rel)
            preds = predict_labels(probs)
            for m in ("macro_f1", "accuracy"):
                ci = multi_model_bootstrap(y, preds, grp, ALL6, metric=m, n_boot=1000)
                row[f"test_{m}_low"], row[f"test_{m}_high"] = ci["low"], ci["high"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(folds: pd.DataFrame, backbones, variant: str, reference: str = "none") -> pd.DataFrame:
    """variant vs reference on identical folds.

    CV: Nadeau-Bengio corrected resampled t-test and Wilcoxon signed-rank on the
    15 paired validation-fold scores.  Test: Wilcoxon over the 15 paired fold
    models and photo-level paired bootstrap of the mean macro-F1 difference.
    """
    sp_n_train = int(np.mean([len(f["train"]) for f in _splits()["public_cv"]["1"]]))
    sp_n_val = int(np.mean([len(f["val"]) for f in _splits()["public_cv"]["1"]]))
    rows = []
    for b in backbones:
        a = folds[(folds.backbone == b) & (folds.variant == variant)].set_index(["repeat", "fold"])
        r = folds[(folds.backbone == b) & (folds.variant == reference)].set_index(["repeat", "fold"])
        idx = a.index.intersection(r.index)
        if len(idx) < 3:
            continue
        d_cv = (a.loc[idx, "cv_macro_f1"] - r.loc[idx, "cv_macro_f1"]).to_numpy()
        tt = corrected_resampled_ttest(d_cv, sp_n_train, sp_n_val)
        wc = wilcoxon_paired(a.loc[idx, "cv_macro_f1"], r.loc[idx, "cv_macro_f1"])
        wt = wilcoxon_paired(a.loc[idx, "test_macro_f1"], r.loc[idx, "test_macro_f1"])
        pa, rel = test_predictions(b, variant)
        pr, _ = test_predictions(b, reference)
        boot = paired_multi_bootstrap(labels_of(rel), predict_labels(pa), predict_labels(pr),
                                      groups_of(rel), ALL6, n_boot=1000)
        rows.append({"backbone": b, "variant": variant, "n_pairs": len(idx),
                     "cv_ref": r.loc[idx, "cv_macro_f1"].mean(), "cv_var": a.loc[idx, "cv_macro_f1"].mean(),
                     "cv_delta": tt["mean"], "cv_delta_sd": tt["sd"], "cv_p_nb": tt["p_value"],
                     "cv_p_wilcoxon": wc["p_value"],
                     "test_ref": r.loc[idx, "test_macro_f1"].mean(), "test_var": a.loc[idx, "test_macro_f1"].mean(),
                     "test_delta": boot["diff"], "test_delta_low": boot["low"], "test_delta_high": boot["high"],
                     "test_p_boot": boot["p_value"], "test_p_wilcoxon": wt["p_value"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["cv_p_nb_holm"] = holm_correction(out["cv_p_nb"])
        out["test_p_boot_holm"] = holm_correction(out["test_p_boot"])
    return out


def _splits():
    from .common import splits
    return splits()


def per_class(backbones, variants) -> pd.DataFrame:
    rows = []
    for b in backbones:
        for v in variants:
            probs, rel = test_predictions(b, v)
            if probs is None:
                continue
            y = labels_of(rel)
            dfs = [per_class_from_cm(confusion(y, p), ALL6) for p in predict_labels(probs)]
            cat = pd.concat(dfs)
            agg = cat.groupby("class", sort=False).agg(support=("support", "first"),
                                                      precision=("precision", "mean"),
                                                      recall=("recall", "mean"), f1=("f1", "mean"),
                                                      f1_sd=("f1", "std")).reset_index()
            agg.insert(0, "variant", v)
            agg.insert(0, "backbone", b)
            rows.append(agg)
    return pd.concat(rows) if rows else pd.DataFrame()


def mean_confusion(backbone: str, variant: str, key: str = "public_test", allowed=ALL6) -> np.ndarray:
    probs, rel = test_predictions(backbone, variant, key)
    y = labels_of(rel)
    return np.mean([confusion(y, p) for p in predict_labels(probs, allowed)], axis=0)


def run() -> dict:
    d = out_dir("source")
    folds = collect_source()
    if folds.empty:
        print("no source runs yet")
        return {}
    folds.to_csv(d / "source_fold_metrics.csv", index=False)
    summ = summary_table(folds)
    summ.to_csv(d / "source_summary.csv", index=False)
    cbam = paired_comparison(folds, C.BACKBONES, "cbam")
    cbam.to_csv(d / "cbam_vs_base.csv", index=False)
    abl = pd.concat([paired_comparison(folds, C.ABLATION_BACKBONES, v)
                     for v in ["cbam"] + C.ABLATION_VARIANTS], ignore_index=True)
    if not abl.empty:
        abl["cv_p_nb_holm"] = holm_correction(abl["cv_p_nb"])
    abl.to_csv(d / "ablation.csv", index=False)
    pc = per_class(C.BACKBONES, C.MAIN_VARIANTS)
    pc.to_csv(d / "source_per_class.csv", index=False)
    return {"folds": folds, "summary": summ, "cbam": cbam, "ablation": abl, "per_class": pc}
