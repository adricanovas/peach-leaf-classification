"""Builds the manifest and every grouped split used in the paper.

Run once (``python -m peachleaf.splits``); the outputs in ``splits/`` are
committed and released with the code so the exact partitions can be reproduced.

Protocol
--------
Public domain (6 classes)
    * hold-out test: one of five StratifiedGroupKFold folds (~20 %), fixed for
      every model.  Among the five candidate folds we keep the one whose class
      proportions are closest (L1) to the full dataset.
    * development set: the remaining ~80 %, split with StratifiedGroupKFold
      (5 folds) repeated 3 times with different shuffles -> 15 (train, val)
      pairs.  Validation folds drive early stopping and are used for the
      cross-validation estimates.
Local domain (4 classes)
    * 5-fold StratifiedGroupKFold repeated 3 times -> 15 (train, test) pairs.
      Within each training part a grouped 20 % inner validation split is
      reserved for early stopping and model/strategy selection, so the test
      fold is never used for any decision.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from .data import build_manifest


def _sgkf(df: pd.DataFrame, n_splits: int, seed: int):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(df, df["label_idx"], df["group"]))


def _rel(df: pd.DataFrame, idx) -> list:
    return df.iloc[idx]["relpath"].tolist()


def build_splits(manifest: pd.DataFrame) -> dict:
    pub = manifest[manifest.domain == "public"].reset_index(drop=True)
    loc = manifest[manifest.domain == "local"].reset_index(drop=True)

    # ---- public hold-out test ------------------------------------------------
    global_p = pub["label_idx"].value_counts(normalize=True).sort_index()
    best, best_d = None, np.inf
    for _, te in _sgkf(pub, C.PUBLIC_TEST_FRACTION_FOLDS, C.SPLIT_SEED):
        p = pub.iloc[te]["label_idx"].value_counts(normalize=True)
        p = p.reindex(global_p.index, fill_value=0.0)
        d = float(np.abs(p - global_p).sum())
        if d < best_d:
            best, best_d = te, d
    test_mask = np.zeros(len(pub), bool)
    test_mask[best] = True
    dev = pub[~test_mask].reset_index(drop=True)

    splits = {
        "protocol_version": 2,
        "seed": C.SPLIT_SEED,
        "public_test": _rel(pub, np.where(test_mask)[0]),
        "public_dev": dev["relpath"].tolist(),
        "public_cv": {},
        "local_all": loc["relpath"].tolist(),
        "local_cv": {},
    }

    # ---- public development CV --------------------------------------------------
    for r in range(1, C.CV_REPEATS + 1):
        folds = []
        for tr, va in _sgkf(dev, C.CV_SPLITS, C.SPLIT_SEED + r):
            folds.append({"train": _rel(dev, tr), "val": _rel(dev, va)})
        splits["public_cv"][str(r)] = folds

    # ---- local CV with inner validation ------------------------------------------
    for r in range(1, C.LOCAL_REPEATS + 1):
        folds = []
        for k, (trv, te) in enumerate(_sgkf(loc, C.LOCAL_SPLITS, C.SPLIT_SEED + 100 + r)):
            trv_df = loc.iloc[trv].reset_index(drop=True)
            n_inner = int(round(1 / C.INNER_VAL_FRACTION))
            tr, va = _sgkf(trv_df, n_inner, C.SPLIT_SEED + 1000 * r + k)[0]
            folds.append({"train": _rel(trv_df, tr), "val": _rel(trv_df, va),
                          "test": _rel(loc, te)})
        splits["local_cv"][str(r)] = folds
    return splits


def check_disjoint(manifest: pd.DataFrame, splits: dict) -> None:
    g = manifest.set_index("relpath")["group"]

    def groups(rel):
        return set(g.loc[rel])

    assert not groups(splits["public_test"]) & groups(splits["public_dev"])
    for folds in splits["public_cv"].values():
        for f in folds:
            assert not groups(f["train"]) & groups(f["val"])
            assert not (groups(f["train"]) | groups(f["val"])) & groups(splits["public_test"])
    for folds in splits["local_cv"].values():
        tests = []
        for f in folds:
            a, b, c = groups(f["train"]), groups(f["val"]), groups(f["test"])
            assert not (a & b) and not (a & c) and not (b & c)
            tests += f["test"]
        assert sorted(tests) == sorted(splits["local_all"]), "local folds must cover all leaves"


def summary(manifest: pd.DataFrame, splits: dict) -> pd.DataFrame:
    lab = manifest.set_index("relpath")["label"]
    rows = []

    def add(name, rel):
        vc = lab.loc[rel].value_counts()
        rows.append({"split": name, "n": len(rel),
                     "photos": manifest.set_index("relpath").loc[rel, "group"].nunique(),
                     **{c: int(vc.get(c, 0)) for c in C.CLASS_NAMES}})

    add("public_all", manifest[manifest.domain == "public"].relpath.tolist())
    add("public_dev", splits["public_dev"])
    add("public_test", splits["public_test"])
    for r, folds in splits["public_cv"].items():
        for k, f in enumerate(folds, 1):
            add(f"public_cv_r{r}_f{k}_val", f["val"])
    add("local_all", splits["local_all"])
    for r, folds in splits["local_cv"].items():
        for k, f in enumerate(folds, 1):
            add(f"local_r{r}_f{k}_train", f["train"])
            add(f"local_r{r}_f{k}_val", f["val"])
            add(f"local_r{r}_f{k}_test", f["test"])
    return pd.DataFrame(rows)


def main():
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    manifest.to_csv(C.MANIFEST_CSV, index=False)
    splits = build_splits(manifest)
    check_disjoint(manifest, splits)
    with (C.SPLITS_DIR / "splits.json").open("w") as f:
        json.dump(splits, f, indent=1)
    s = summary(manifest, splits)
    s.to_csv(C.SPLITS_DIR / "split_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
