"""Leaves shown in Figure 1: typical leaves of every class in each domain.

A leaf is "typical" when the 11 baseline backbones (15 fold models each) agree on
its class: leaves are ranked by the mean probability assigned to their true class
(public hold-out; orchard zero-shot with the two absent classes masked), at most
one leaf per photograph and no two near-identical leaves, and the top three are shown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .common import LOCAL4, groups_of, labels_of, out_dir
from .source import test_predictions

N_PER_CLASS = 3


def typicality(key: str, allowed) -> pd.DataFrame:
    P, rel = [], None
    for b in C.BACKBONES:
        p, rel = test_predictions(b, "none", key)
        P.append(p)
    P = np.concatenate(P)
    mask = np.zeros(C.NUM_CLASSES, bool)
    mask[list(allowed)] = True
    P = np.where(mask, P, 0.0)
    P = P / P.sum(-1, keepdims=True)
    y = labels_of(rel)
    return pd.DataFrame({"relpath": rel, "label": y, "group": groups_of(rel),
                         "conf": P[:, np.arange(len(y)), y].mean(0)})


def _similar(a: str, b: str, min_inliers: int = 20) -> bool:
    """Near-identical leaves (SIFT matches consistent with one homography); avoids look-alike panels."""
    import cv2

    sift, imgs = cv2.SIFT_create(4000), []
    for rp in (a, b):
        g = cv2.imread(str(C.DATA_ROOT / rp), cv2.IMREAD_GRAYSCALE)
        imgs.append(sift.detectAndCompute(g, (g > 0).astype(np.uint8)))
    (k1, d1), (k2, d2) = imgs
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return False
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.8 * n.distance]
    if len(good) < min_inliers:
        return False
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    return mask is not None and int(mask.sum()) >= min_inliers


def pick(df: pd.DataFrame) -> dict:
    out = {}
    for c, x in df.groupby("label"):
        chosen = []
        for rp in x.sort_values("conf", ascending=False).drop_duplicates("group").relpath:
            if not any(_similar(rp, q) for q in chosen):
                chosen.append(rp)
            if len(chosen) == N_PER_CLASS:
                break
        out[int(c)] = chosen
    return out


def run():
    pub = pick(typicality("public_test", range(C.NUM_CLASSES)))
    loc = pick(typicality("local_all", LOCAL4))
    pd.DataFrame([{"domain": d, "class": C.CLASS_NAMES[c], "rank": i + 1, "relpath": p}
                  for d, sel in (("public", pub), ("orchard", loc)) for c, ps in sel.items()
                  for i, p in enumerate(ps)]).to_csv(out_dir("source") / "typical_leaves.csv", index=False)


if __name__ == "__main__":
    run()
