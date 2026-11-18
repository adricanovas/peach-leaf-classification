"""Cost aggregation and the CSV files behind the figures of the paper.

    python -m peachleaf.analysis            # everything available under runs/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..models import pretty_name
from .common import LOCAL4, confusion, labels_of, out_dir, predict_labels, splits
from .source import mean_confusion


# =============================================================================
# Cost
# =============================================================================

def cost_table() -> pd.DataFrame:
    rows = []
    for f in (C.RUNS_ROOT / "cost").glob("*/*.json"):
        r = json.loads(f.read_text())
        r["setting"] = f.parent.name
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["backbone", "variant", "params", "attention_params", "flops", "file_mb"]
    wide = df[df.setting == df.setting.iloc[0]][keep].drop_duplicates(["backbone", "variant"])
    for s, g in df.groupby("setting"):
        g = g.set_index(["backbone", "variant"])
        for c in ("bs1_latency_ms_median", "bs1_latency_ms_p95", "bs32_throughput_img_s", "bs1_peak_gpu_mb"):
            wide = wide.join(g[c].rename(f"{s}_{c}"), on=["backbone", "variant"])
    base = wide[wide.variant == "none"].set_index("backbone")
    for c in ("params", "flops") + tuple(x for x in wide.columns if x.endswith("latency_ms_median")):
        wide[f"{c}_overhead_pct"] = [100 * (r[c] / base.loc[r.backbone, c] - 1) if r.backbone in base.index else np.nan
                                     for _, r in wide.iterrows()]
    return wide


def confusion_long(cm: np.ndarray, model: str, labels, setting: str = "") -> pd.DataFrame:
    rows = [{"model": model, "setting": setting, "true": labels[i], "pred": labels[j], "mean_count": cm[i, j]}
            for i in range(len(labels)) for j in range(len(labels))]
    return pd.DataFrame(rows)


def run_all():
    from . import examples, pooled, source, target

    cost = cost_table()
    if not cost.empty:
        cost.to_csv(out_dir("cost") / "cost.csv", index=False)
    src = source.run()
    if src:
        summ = src["summary"]
        reps = [(b, v) for b, v in (("efficientnetb0", "none"), ("densenet121", "none"),
                                    ("efficientnetb5", "none"), ("efficientnetb5", "cbam"))
                if ((summ.backbone == b) & (summ.variant == v)).any()]
        if reps:
            pd.concat([confusion_long(mean_confusion(b, v), pretty_name(b, v), C.CLASS_NAMES) for b, v in reps]
                      ).to_csv(out_dir("source") / "confusion_public.csv", index=False)
    tgt = target.run(src.get("summary") if src else None)
    if not tgt["adaptation"].empty:
        y = labels_of(np.asarray(splits()["local_all"]))
        cms = []
        for b, v in (("efficientnetb3", "cbam"), ("densenet121", "none")):
            for m, setting in (("zs", "zero-shot"), ("selected", "adapted (nested)")):
                P = tgt["store"].get((b, v), {}).get(m)
                if P is not None:
                    cm = np.mean([confusion(y, p) for p in predict_labels(P, LOCAL4)], axis=0)
                    cms.append(confusion_long(cm[np.ix_(LOCAL4, LOCAL4)], pretty_name(b, v), C.LOCAL_CLASSES, setting))
        if cms:
            pd.concat(cms).to_csv(out_dir("target") / "confusion_orchard.csv", index=False)
        pooled.run()
    examples.run()
    print("reports written to", C.REPORTS_ROOT)
