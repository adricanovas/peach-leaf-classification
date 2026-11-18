"""Image-level characterisation of the public -> orchard domain shift.

Leaves are polygon-masked crops on a black background, so every colour/texture
statistic is computed on leaf pixels only.  Run on CPU::

    python -m peachleaf.domain_stats
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as C
from .data import load_manifest
from .stats import holm_correction

STAT_COLS = ["height", "width", "elongation", "megapixels", "background_fraction", "mean_R", "mean_G",
             "mean_B", "brightness", "saturation", "hue_circular_mean", "colorfulness", "contrast",
             "sharpness", "excess_green", "edge_compactness"]


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], np.sort(b[~np.isnan(b)])
    if not len(a) or not len(b):
        return float("nan")
    gt = np.searchsorted(b, a, side="left").sum()
    lt = (len(b) - np.searchsorted(b, a, side="right")).sum()
    return float((gt - lt) / (len(a) * len(b)))


def image_stats(path: Path) -> dict:
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    out = {"height": h, "width": w, "elongation": max(h, w) / min(h, w), "megapixels": h * w / 1e6}
    mask = rgb.max(axis=2) > 10
    out["background_fraction"] = 1.0 - mask.mean()
    if not mask.any():
        return out
    R, G, B = (rgb[..., i][mask].astype(float) for i in range(3))
    out.update(mean_R=R.mean(), mean_G=G.mean(), mean_B=B.mean())
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = np.deg2rad(hsv[..., 0][mask].astype(float) * 2.0)
    out["hue_circular_mean"] = float(np.rad2deg(np.arctan2(np.sin(hue).mean(), np.cos(hue).mean())) % 360)
    out["saturation"] = hsv[..., 1][mask].mean()
    out["brightness"] = hsv[..., 2][mask].mean()
    rg, yb = R - G, 0.5 * (R + G) - B
    out["colorfulness"] = math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out["contrast"] = gray[mask].astype(float).std()
    # sharpness on a common scale (longest side 512 px), inside the eroded leaf
    s = 512.0 / max(h, w)
    g2 = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    m2 = cv2.resize(mask.astype(np.uint8), g2.shape[::-1], interpolation=cv2.INTER_NEAREST)
    m2 = cv2.erode(m2, np.ones((5, 5), np.uint8)) > 0
    lap = cv2.Laplacian(g2, cv2.CV_64F)
    out["sharpness"] = float(lap[m2].var()) if m2.any() else float("nan")
    out["excess_green"] = float(((2 * G - R - B) / (R + G + B + 1e-6)).mean())
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        out["edge_compactness"] = cv2.arcLength(c, True) ** 2 / (4 * np.pi * a) if a > 0 else float("nan")
    return out


def compute_image_stats(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = [{**r._asdict(), **image_stats(C.DATA_ROOT / r.relpath)}
            for r in manifest[["relpath", "domain", "label", "group", "campaign"]].itertuples(index=False)]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows, pvals = [], []
    pub, loc = df[df.domain == "public"], df[df.domain == "local"]
    for col in STAT_COLS:
        a, b = pub[col].to_numpy(float), loc[col].to_numpy(float)
        row = {"stat": col}
        for name, sub in [("public", pub), ("local", loc)] + [
                (f"local_{c}", loc[loc.campaign.astype(str) == c]) for c in sorted(loc.campaign.astype(str).unique())]:
            v = sub[col].dropna()
            row[f"{name}_median"] = v.median()
            row[f"{name}_iqr"] = v.quantile(0.75) - v.quantile(0.25)
        row["cliffs_delta_local_vs_public"] = cliffs_delta(b, a)
        p = sps.mannwhitneyu(a[~np.isnan(a)], b[~np.isnan(b)]).pvalue
        row["p_mannwhitney"] = p
        pvals.append(p)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(pvals)
    return out.reindex(out["cliffs_delta_local_vs_public"].abs().sort_values(ascending=False).index)


def main():
    out = C.REPORTS_ROOT / "domain_shift"
    out.mkdir(parents=True, exist_ok=True)
    df = compute_image_stats(load_manifest())
    df.to_csv(out / "image_stats.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "domain_stats_summary.csv", index=False)
    print(s[["stat", "public_median", "local_median", "cliffs_delta_local_vs_public", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
