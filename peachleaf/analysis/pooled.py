"""Effects pooled over backbones: one estimate per comparison instead of one test per backbone.

CV: the paired difference is averaged over backbones within each of the 15
(repeat, fold) splits and tested with the Nadeau-Bengio corrected t-test.
Fixed evaluation sets (public hold-out, orchard): the prediction vectors of all
backbones are stacked and compared with the photo-level paired bootstrap of the
mean macro-F1, so the shared test photographs are resampled once for all models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as sps

from .. import config as C
from ..stats import corrected_resampled_ttest
from .common import ALL6, LOCAL4, evaluate, groups_of, labels_of, out_dir, paired_multi_bootstrap, predict_labels
from .source import _splits, collect_source, test_predictions
from .target import MAIN, STRATS, _assemble, _local_order, _selected

N_BOOT = 2000


def _stack(blocks):
    return np.concatenate([np.atleast_2d(b) for b in blocks], axis=0)


def _cv_pooled(folds: pd.DataFrame, backbones, variant: str, reference: str = "none") -> dict:
    n_train = int(np.mean([len(f["train"]) for f in _splits()["public_cv"]["1"]]))
    n_val = int(np.mean([len(f["val"]) for f in _splits()["public_cv"]["1"]]))
    f = folds[folds.backbone.isin(backbones)]
    a = f[f.variant == variant].set_index(["backbone", "repeat", "fold"])["cv_macro_f1"]
    r = f[f.variant == reference].set_index(["backbone", "repeat", "fold"])["cv_macro_f1"]
    d = (a - r).dropna().groupby(level=["repeat", "fold"]).mean()
    tt = corrected_resampled_ttest(d.to_numpy(), n_train, n_val)
    se = tt["sd"] * np.sqrt(1 / len(d) + n_val / n_train)
    half = se * sps.t.ppf(0.975, len(d) - 1)
    return {"diff": tt["mean"], "low": tt["mean"] - half, "high": tt["mean"] + half, "p_value": tt["p_value"]}


def _fixed_pooled(key: str, backbones, variant: str, reference: str, labels) -> dict:
    A, B, rel0 = [], [], None
    for b in backbones:
        pa, rel = test_predictions(b, variant, key)
        pr, rel_r = test_predictions(b, reference, key)
        if pa is None or pr is None:
            continue
        assert (rel == rel_r).all() and (rel0 is None or (rel == rel0).all())
        rel0 = rel
        A.append(predict_labels(pa, labels))
        B.append(predict_labels(pr, labels))
    return paired_multi_bootstrap(labels_of(rel0), _stack(A), _stack(B), groups_of(rel0), labels, n_boot=N_BOOT)


def attention_effects() -> pd.DataFrame:
    folds = collect_source()
    rows = []
    for variant, backbones in [("cbam", C.BACKBONES)] + [(v, C.ABLATION_BACKBONES)
                                                          for v in ["cbam"] + C.ABLATION_VARIANTS]:
        n = len(backbones)
        rows.append({"comparison": f"{variant} - none", "backbones": n, "set": "public CV",
                     **_cv_pooled(folds, backbones, variant)})
        rows.append({"comparison": f"{variant} - none", "backbones": n, "set": "public hold-out",
                     **_fixed_pooled("public_test", backbones, variant, "none", ALL6)})
        rows.append({"comparison": f"{variant} - none", "backbones": n, "set": "orchard zero-shot",
                     **_fixed_pooled("local_all", backbones, variant, "none", LOCAL4)})
    return pd.DataFrame(rows)


def adaptation_effects(adaptation_csv: pd.DataFrame) -> pd.DataFrame:
    rel_all, _ = _local_order()
    y, g = labels_of(rel_all), groups_of(rel_all)
    store = {}
    for b, v in MAIN:
        m = {s: _assemble("adapt", b, v, s) for s in ["zs"] + STRATS}
        m["selected"] = _selected(b, v)[0]
        store[(b, v)] = {s: predict_labels(P, LOCAL4) for s, P in m.items() if P is not None}
    pairs = [("fta", "zs"), ("lpft", "zs"), ("selected", "zs"), ("ftl", "lp"), ("fta", "lp"),
             ("lpft", "lp"), ("fta", "ftl"), ("lpft", "fta")]
    rows = []
    for variants in (C.MAIN_VARIANTS, ["none"], ["cbam"]):
        keys = [k for k in store if k[1] in variants]
        for a, b in pairs:
            ks = [k for k in keys if a in store[k] and b in store[k]]
            t = paired_multi_bootstrap(y, _stack([store[k][a] for k in ks]), _stack([store[k][b] for k in ks]),
                                       g, LOCAL4, n_boot=N_BOOT)
            # source retention (public hold-out, four shared classes); models as units
            ad = adaptation_csv[adaptation_csv.variant.isin(variants)].pivot_table(
                index=["backbone", "variant"], columns="method", values="pub4")
            ret = (ad[a] - ad[b]).dropna() if a in ad and b in ad else pd.Series(dtype=float)
            rows.append({"comparison": f"{a} - {b}", "variants": "+".join(variants), "models": len(ks),
                         "n_positive": int(sum(_mf1(y, store[k][a]) > _mf1(y, store[k][b]) for k in ks)),
                         **t, "pub4_diff": ret.mean() if len(ret) else np.nan})
    return pd.DataFrame(rows)


def _mf1(y, preds) -> float:
    return float(np.mean([evaluate(y, p, LOCAL4)["macro_f1"] for p in preds]))


def run() -> dict:
    d = out_dir("pooled")
    att = attention_effects()
    att.to_csv(d / "attention_pooled.csv", index=False)
    ad = pd.read_csv(C.REPORTS_ROOT / "target" / "adaptation.csv")
    adp = adaptation_effects(ad)
    adp.to_csv(d / "adaptation_pooled.csv", index=False)
    return {"attention": att, "adaptation": adp}


if __name__ == "__main__":
    out = run()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for k, v in out.items():
            print(k)
            print(v.round(4).to_string(index=False))
