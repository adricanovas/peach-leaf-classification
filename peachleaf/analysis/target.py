"""Target-domain results: zero-shot transfer, domain characterisation and adaptation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config as C
from ..stats import holm_correction, spearman_ci
from .common import (ALL6, LOCAL4, confusion, evaluate, groups_of, labels_of, load_unit, manifest,
                     metrics_from_cm, multi_model_bootstrap, out_dir, paired_multi_bootstrap,
                     per_class_from_cm, predict_labels, splits)
from .source import test_predictions

MAIN = [(b, v) for b in C.BACKBONES for v in C.MAIN_VARIANTS]
STRATS = ["lp", "ftl", "fta", "lpft"]


def _local_order() -> Tuple[np.ndarray, Dict[str, int]]:
    rel = np.asarray(splits()["local_all"])
    return rel, {r: i for i, r in enumerate(rel)}


# =============================================================================
# Zero-shot transfer
# =============================================================================

def _two_sample_gap(y_s, P_s, g_s, y_t, P_t, g_t, labels, n_boot=1000, seed=0):
    """CI of mean_metric(target) - mean_metric(source) resampling photos in each domain."""
    rng = np.random.default_rng(seed)

    def members(g):
        u, inv = np.unique(g, return_inverse=True)
        return [np.flatnonzero(inv == i) for i in range(len(u))]

    ms, mt = members(g_s), members(g_t)

    def mean_metric(y, P, idx):
        return np.mean([metrics_from_cm(confusion(y[idx], p[idx]), labels)["macro_f1"] for p in P])

    point = mean_metric(y_t, P_t, np.arange(len(y_t))) - mean_metric(y_s, P_s, np.arange(len(y_s)))
    d = []
    for _ in range(n_boot):
        i_s = np.concatenate([ms[i] for i in rng.integers(0, len(ms), len(ms))])
        i_t = np.concatenate([mt[i] for i in rng.integers(0, len(mt), len(mt))])
        d.append(mean_metric(y_t, P_t, i_t) - mean_metric(y_s, P_s, i_s))
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def zero_shot() -> pd.DataFrame:
    rows = []
    for b, v in MAIN:
        P_loc, rel_loc = test_predictions(b, v, "local_all")
        P_pub, rel_pub = test_predictions(b, v, "public_test")
        if P_loc is None:
            continue
        y_loc, g_loc = labels_of(rel_loc), groups_of(rel_loc)
        y_pub, g_pub = labels_of(rel_pub), groups_of(rel_pub)
        shared = np.isin(y_pub, LOCAL4)
        row = {"backbone": b, "variant": v, "n_models": len(P_loc)}
        # source performance: all six classes, and restricted to the four shared classes
        src6 = [evaluate(y_pub, p, ALL6)["macro_f1"] for p in predict_labels(P_pub)]
        P_pub4 = predict_labels(P_pub[:, shared], LOCAL4)
        src4 = [evaluate(y_pub[shared], p, LOCAL4) for p in P_pub4]
        row["src6_macro_f1"] = np.mean(src6)
        row["src4_macro_f1"] = np.mean([m["macro_f1"] for m in src4])
        row["src4_accuracy"] = np.mean([m["accuracy"] for m in src4])
        # target: masked to the four orchard classes (deployment with a known label set)
        P_loc4 = predict_labels(P_loc, LOCAL4)
        tgt = pd.DataFrame([evaluate(y_loc, p, LOCAL4) for p in P_loc4])
        for m in ["macro_f1", "accuracy", "balanced_accuracy", "mcc", "false_alarm_rate", "miss_rate"]:
            row[f"zs_{m}"] = tgt[m].mean()
            row[f"zs_{m}_sd"] = tgt[m].std(ddof=1)
        ci = multi_model_bootstrap(y_loc, P_loc4, g_loc, LOCAL4, n_boot=1000)
        row["zs_macro_f1_low"], row["zs_macro_f1_high"] = ci["low"], ci["high"]
        # target without masking: predictions of absent classes count as errors
        P_loc6 = predict_labels(P_loc)
        row["zs6_macro_f1"] = np.mean([evaluate(y_loc, p, LOCAL4)["macro_f1"] for p in P_loc6])
        row["zs6_absent_rate"] = float(np.mean(~np.isin(P_loc6, LOCAL4)))
        gap, lo, hi = _two_sample_gap(y_pub[shared], P_pub4, g_pub[shared], y_loc, P_loc4, g_loc, LOCAL4)
        row.update(gap4=gap, gap4_low=lo, gap4_high=hi, gap4_rel=gap / row["src4_macro_f1"])
        # Label-free BatchNorm adaptation of the five repeat-1 models: classic AdaBN
        # (statistics re-estimated on orchard images) and moment alignment
        bn = {"adabn": ([], []), "align": ([], [])}
        for k in range(1, C.CV_SPLITS + 1):
            u = load_unit("adabn", b, v, 1, k)
            if u is not None and "align/local_all" in u:
                for meth, (L, Pb) in bn.items():
                    if f"{meth}/local_all" in u:
                        L.append(u[f"{meth}/local_all"][0])
                        Pb.append(u[f"{meth}/public_test"][0])
        A, A_pub = bn["adabn"]
        if A:
            ad = [evaluate(y_loc, p, LOCAL4)["macro_f1"] for p in predict_labels(np.stack(A), LOCAL4)]
            row["adabn_macro_f1"], row["adabn_macro_f1_sd"] = np.mean(ad), np.std(ad, ddof=1) if len(ad) > 1 else 0.0
            row["adabn_src6_macro_f1"] = np.mean([evaluate(y_pub, p, ALL6)["macro_f1"]
                                                  for p in predict_labels(np.stack(A_pub))])
        Al, Ap = bn["align"]
        if Al:
            al = [evaluate(y_loc, p, LOCAL4)["macro_f1"] for p in predict_labels(np.stack(Al), LOCAL4)]
            row["align_macro_f1"], row["align_macro_f1_sd"] = np.mean(al), np.std(al, ddof=1) if len(al) > 1 else 0.0
            row["align_src6_macro_f1"] = np.mean([evaluate(y_pub, p, ALL6)["macro_f1"]
                                                  for p in predict_labels(np.stack(Ap))])
            # zero-shot of the same five models, for a paired comparison
            P5 = np.stack([load_unit("source", b, v, 1, k)["local_all"][0]
                           for k in range(1, C.CV_SPLITS + 1) if load_unit("adabn", b, v, 1, k) is not None])
            t = paired_multi_bootstrap(y_loc, predict_labels(np.stack(Al), LOCAL4), predict_labels(P5, LOCAL4),
                                       g_loc, LOCAL4, n_boot=1000)
            row.update(align_vs_zs=t["diff"], align_vs_zs_low=t["low"], align_vs_zs_high=t["high"],
                       align_vs_zs_p=t["p_value"])
        rows.append(row)
    return pd.DataFrame(rows)


def cbam_orchard() -> pd.DataFrame:
    """Paired CBAM - baseline difference in zero-shot orchard macro-F1 (same 182 leaves)."""
    rows = []
    for b in C.BACKBONES:
        A, rel = test_predictions(b, "cbam", "local_all")
        B, rel_b = test_predictions(b, "none", "local_all")
        if A is None or B is None:
            continue
        assert (rel == rel_b).all()
        y, g = labels_of(rel), groups_of(rel)
        t = paired_multi_bootstrap(y, predict_labels(A, LOCAL4), predict_labels(B, LOCAL4), g, LOCAL4,
                                   n_boot=1000)
        rows.append({"backbone": b, "zs_ref": np.mean(
            [evaluate(y, p, LOCAL4)["macro_f1"] for p in predict_labels(B, LOCAL4)]),
            "zs_delta": t["diff"], "zs_delta_low": t["low"], "zs_delta_high": t["high"],
            "zs_p": t["p_value"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["zs_var"] = df["zs_ref"] + df["zs_delta"]
        df["zs_p_holm"] = holm_correction(df["zs_p"])
    return df


def campaign_breakdown(method_preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-campaign recall of each class: tests whether models key on acquisition campaign."""
    rel, _ = _local_order()
    man = manifest().loc[rel]
    y, camp = man["label_idx"].to_numpy(), man["campaign"].astype(str).to_numpy()
    rows = []
    for name, P in method_preds.items():
        P = predict_labels(P, LOCAL4)  # stored as probabilities
        for c in sorted(np.unique(camp)):
            for lbl in LOCAL4:
                m = (camp == c) & (y == lbl)
                if m.sum() == 0:
                    continue
                rec = [(p[m] == lbl).mean() for p in P]
                rows.append({"method": name, "campaign": c, "class": C.CLASS_NAMES[lbl],
                             "n": int(m.sum()), "recall": float(np.mean(rec)),
                             "recall_sd": float(np.std(rec, ddof=1)) if len(rec) > 1 else 0.0})
    return pd.DataFrame(rows)


def rank_agreement(src: pd.DataFrame, zs: pd.DataFrame) -> pd.DataFrame:
    """Does the public benchmark ranking predict orchard behaviour?."""
    m = zs.merge(src[["backbone", "variant", "test_macro_f1_mean"]], on=["backbone", "variant"])
    rows = []
    for x, label in (("test_macro_f1_mean", "public test (6 cl.)"), ("src4_macro_f1", "public test (4 cl.)")):
        for subset, d in (("all", m), ("baselines", m[m.variant == "none"])):
            s = spearman_ci(d[x], d["zs_macro_f1"])
            rows.append({"source_metric": label, "models": subset, **s, "n_models": len(d)})
    return pd.DataFrame(rows)


# =============================================================================
# Adaptation
# =============================================================================

def _assemble(stage: str, b: str, v: str, key: str, part: str = "local_test") -> Optional[np.ndarray]:
    """(R, N_local) predictions: test folds of each repeat concatenated over all orchard leaves."""
    rel_all, pos = _local_order()
    out = []
    for r in range(1, C.LOCAL_REPEATS + 1):
        probs = np.full((len(rel_all), C.NUM_CLASSES), np.nan, np.float32)
        for k in range(1, C.LOCAL_SPLITS + 1):
            u = load_unit(stage, b, v, r, k)
            if u is None or f"{key}/{part}" not in u:
                break
            p, rel = u[f"{key}/{part}"]
            probs[[pos[x] for x in rel]] = p
        else:
            out.append(probs)
    return np.stack(out) if out else None


def _selected(b: str, v: str, candidates=STRATS) -> Optional[np.ndarray]:
    """Nested selection: per fold, the strategy with the best inner-validation macro-F1."""
    rel_all, pos = _local_order()
    out, choices = [], []
    for r in range(1, C.LOCAL_REPEATS + 1):
        probs = np.full((len(rel_all), C.NUM_CLASSES), np.nan, np.float32)
        ok = True
        for k in range(1, C.LOCAL_SPLITS + 1):
            u = load_unit("adapt", b, v, r, k)
            if u is None:
                ok = False
                break
            scores = {}
            for s in candidates:
                pv, rv = u[f"{s}/local_val"]
                scores[s] = evaluate(labels_of(rv), predict_labels(pv, LOCAL4), LOCAL4)["macro_f1"]
            best = max(scores, key=scores.get)
            choices.append(best)
            p, rel = u[f"{best}/local_test"]
            probs[[pos[x] for x in rel]] = p
        if ok:
            out.append(probs)
    return (np.stack(out) if out else None), choices


def _forgetting(b: str, v: str, key: str, stage: str = "adapt") -> Dict[str, float]:
    """Public hold-out macro-F1 after adaptation (6 classes and 4 shared classes)."""
    s6, s4 = [], []
    for r in range(1, C.LOCAL_REPEATS + 1):
        for k in range(1, C.LOCAL_SPLITS + 1):
            u = load_unit(stage, b, v, r, k)
            if u is None or f"{key}/public_test" not in u:
                continue
            p, rel = u[f"{key}/public_test"]
            y = labels_of(rel)
            s6.append(evaluate(y, predict_labels(p), ALL6)["macro_f1"])
            sh = np.isin(y, LOCAL4)
            s4.append(evaluate(y[sh], predict_labels(p[sh], LOCAL4), LOCAL4)["macro_f1"])
    if not s6:
        return {}
    return {"pub6": np.mean(s6), "pub6_sd": np.std(s6, ddof=1), "pub4": np.mean(s4), "pub4_sd": np.std(s4, ddof=1)}


def _summ(P: np.ndarray, y, g, prefix: str) -> Dict[str, float]:
    preds = predict_labels(P, LOCAL4)
    ms = pd.DataFrame([evaluate(y, p, LOCAL4) for p in preds])
    out = {f"{prefix}{m}": ms[m].mean() for m in ["macro_f1", "accuracy", "balanced_accuracy", "mcc",
                                                 "false_alarm_rate", "miss_rate"]}
    out[f"{prefix}macro_f1_sd"] = ms["macro_f1"].std(ddof=1) if len(ms) > 1 else 0.0
    ci = multi_model_bootstrap(y, preds, g, LOCAL4, n_boot=1000)
    out[f"{prefix}macro_f1_low"], out[f"{prefix}macro_f1_high"] = ci["low"], ci["high"]
    ci = multi_model_bootstrap(y, preds, g, LOCAL4, metric="accuracy", n_boot=1000)
    out[f"{prefix}accuracy_low"], out[f"{prefix}accuracy_high"] = ci["low"], ci["high"]
    return out


def adaptation() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    rel_all, _ = _local_order()
    y, g = labels_of(rel_all), groups_of(rel_all)
    rows, tests, store = [], [], {}
    for b, v in MAIN:
        methods = {m: _assemble("adapt", b, v, m) for m in ["zs"] + STRATS}
        methods = {m: P for m, P in methods.items() if P is not None}
        if "zs" not in methods:
            continue
        sel, choices = _selected(b, v)
        if sel is not None:
            methods["selected"] = sel
        store[(b, v)] = methods
        for m, P in methods.items():
            row = {"backbone": b, "variant": v, "method": m, "n_repeats": len(P)}
            row.update(_summ(P, y, g, ""))
            fg = _forgetting(b, v, "selected" if m == "selected" else m) if m != "selected" else {}
            row.update(fg)
            if m == "selected":
                row["selected_counts"] = pd.Series(choices).value_counts().to_dict()
            rows.append(row)
        # paired comparisons on identical leaves (photo-level bootstrap)
        for a, bb in [("fta", "zs"), ("lpft", "zs"), ("fta", "ftl"), ("fta", "lp"),
                      ("lpft", "fta"), ("selected", "zs")]:
            if a in methods and bb in methods:
                t = paired_multi_bootstrap(y, predict_labels(methods[a], LOCAL4),
                                           predict_labels(methods[bb], LOCAL4), g, LOCAL4, n_boot=1000)
                tests.append({"backbone": b, "variant": v, "a": a, "b": bb, **t})
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["p_holm"] = holm_correction(tests["p_value"])
    return pd.DataFrame(rows), tests, store


def baselines(store) -> pd.DataFrame:
    """Local-only (ImageNet), joint public+local and AdaBN vs public-pretrained adaptation."""
    rel_all, _ = _local_order()
    y, g = labels_of(rel_all), groups_of(rel_all)
    rows = []
    for b, v in MAIN:
        cands = {"scratch": _assemble("scratch", b, v, "scratch"),
                 "joint": _assemble("joint", b, v, "joint")}
        for m, P in cands.items():
            if P is None:
                continue
            row = {"backbone": b, "variant": v, "method": m, "n_repeats": len(P), **_summ(P, y, g, "")}
            if m == "joint":
                row.update(_forgetting(b, v, "joint", stage="joint"))
            ref = store.get((b, v), {}).get("fta")
            if ref is not None:
                t = paired_multi_bootstrap(y, predict_labels(P, LOCAL4), predict_labels(ref, LOCAL4),
                                           g, LOCAL4, n_boot=1000)
                row.update({"vs_fta_diff": t["diff"], "vs_fta_low": t["low"], "vs_fta_high": t["high"],
                            "vs_fta_p": t["p_value"]})
            rows.append(row)
    return pd.DataFrame(rows)


def learning_curve() -> pd.DataFrame:
    rel_all, _ = _local_order()
    y, g = labels_of(rel_all), groups_of(rel_all)
    rows = []
    for b in C.CURVE_BACKBONES:
        for n in C.LEARNING_CURVE_SIZES + ["all"]:
            for m in ("fta", "lpft", "scratch"):
                P = _assemble("curve", b, "none", f"{m}_n{n}")
                if P is None:
                    continue
                sizes = []
                for r in range(1, C.LOCAL_REPEATS + 1):
                    for k in range(1, C.LOCAL_SPLITS + 1):
                        u = load_unit("curve", b, "none", r, k)
                        if u is not None:
                            sizes.append(u.meta["subsets"][str(n)]["n_leaves"])
                s = _summ(P, y, g, "")
                rows.append({"backbone": b, "n_target": n, "n_leaves_mean": np.mean(sizes), "method": m,
                             "macro_f1": s["macro_f1"], "low": s["macro_f1_low"], "high": s["macro_f1_high"],
                             "macro_f1_sd": s["macro_f1_sd"]})
    return pd.DataFrame(rows)


def local_per_class(store, method: str = "selected") -> pd.DataFrame:
    rel_all, _ = _local_order()
    y = labels_of(rel_all)
    rows = []
    for (b, v), methods in store.items():
        for m in ("zs", method):
            if m not in methods:
                continue
            cms = [confusion(y, p) for p in predict_labels(methods[m], LOCAL4)]
            pc = per_class_from_cm(np.mean(cms, axis=0).round().astype(int), LOCAL4)
            pc.insert(0, "method", m)
            pc.insert(0, "variant", v)
            pc.insert(0, "backbone", b)
            rows.append(pc)
    return pd.concat(rows) if rows else pd.DataFrame()


def run(src_summary: Optional[pd.DataFrame] = None) -> dict:
    d = out_dir("target")
    res = {}
    zs = zero_shot()
    zs.to_csv(d / "zero_shot.csv", index=False)
    res["zero_shot"] = zs
    if src_summary is not None and not zs.empty:
        ra = rank_agreement(src_summary, zs)
        ra.to_csv(d / "rank_agreement.csv", index=False)
        res["rank"] = ra
    co = cbam_orchard()
    co.to_csv(d / "cbam_orchard.csv", index=False)
    res["cbam_orchard"] = co
    ad, tests, store = adaptation()
    ad.to_csv(d / "adaptation.csv", index=False)
    tests.to_csv(d / "adaptation_tests.csv", index=False)
    res.update(adaptation=ad, adaptation_tests=tests, store=store)
    camp = campaign_breakdown({f"{b}{'+cbam' if v == 'cbam' else ''}|{m}": P
                               for (b, v), ms in store.items() for m, P in ms.items()
                               if m in ("zs", "selected")})
    camp.to_csv(d / "campaign_recall.csv", index=False)
    res["campaign"] = camp
    for name, fn in (("learning_curve", learning_curve),):
        df = fn()
        df.to_csv(d / f"{name}.csv", index=False)
        res[name] = df
    bl = baselines(store)
    bl.to_csv(d / "baselines.csv", index=False)
    res["baselines"] = bl
    pc = local_per_class(store)
    pc.to_csv(d / "local_per_class.csv", index=False)
    res["local_per_class"] = pc
    return res
