"""Generates the lists of pending experimental units (stdlib only).

    python3 -m peachleaf.tasks source_main          # writes tasks/source_main.txt
    python3 -m peachleaf.tasks --list               # pending/total per task group

Each line holds the arguments of one ``python -m peachleaf.run`` call.  Units
whose ``DONE`` marker already exists are skipped, so re-submitting a group
only runs what is missing (resumable after time-outs or node failures).
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from . import config as C

REPEATS = range(1, C.CV_REPEATS + 1)
FOLDS = range(1, C.CV_SPLITS + 1)
LREPEATS = range(1, C.LOCAL_REPEATS + 1)
LFOLDS = range(1, C.LOCAL_SPLITS + 1)


def _cv(stage, backbones, variants, repeats, folds):
    return [(stage, b, v, r, k) for b, v, r, k in product(backbones, variants, repeats, folds)]


GROUPS = {
    # public-domain cross-validation (main comparison and attention ablations)
    "source_main": lambda: _cv("source", C.BACKBONES, C.MAIN_VARIANTS, REPEATS, FOLDS),
    "source_ablation": lambda: _cv("source", C.ABLATION_BACKBONES, C.ABLATION_VARIANTS, REPEATS, FOLDS),
    # target-domain stages (need the repeat-1 source models)
    "adapt": lambda: _cv("adapt", C.BACKBONES, C.MAIN_VARIANTS, LREPEATS, LFOLDS),
    "adabn": lambda: _cv("adabn", C.BACKBONES, C.MAIN_VARIANTS, [1], FOLDS),
    "scratch": lambda: _cv("scratch", C.BACKBONES, ["none"], LREPEATS, LFOLDS),
    "joint": lambda: _cv("joint", C.JOINT_BACKBONES, C.MAIN_VARIANTS, LREPEATS, LFOLDS),
    "curve": lambda: _cv("curve", C.CURVE_BACKBONES, ["none"], LREPEATS, LFOLDS),
    "cost_gpu": lambda: _cost("gpu", 0),
    "cost_cpu": lambda: _cost("cpu", 4),
}


def _cost(device, threads):
    pairs = list(product(C.BACKBONES, C.MAIN_VARIANTS)) + \
        list(product(C.ABLATION_BACKBONES, C.ABLATION_VARIANTS))
    return [("cost", b, v, 1, 1, device, threads) for b, v in pairs]


def is_done(task) -> bool:
    stage, b, v, r, k = task[:5]
    if stage == "cost":
        device, threads = task[5:]
        return (C.RUNS_ROOT / "cost" / f"{device}{threads or ''}" / f"{b}__{v}.json").exists()
    return (C.RUNS_ROOT / stage / f"{b}__{v}" / f"r{r}_f{k}" / "DONE").exists()


def to_args(task) -> str:
    stage, b, v, r, k = task[:5]
    s = f"{stage} --backbone {b} --variant {v} --repeat {r} --fold {k}"
    if stage == "cost":
        s += f" --device {task[5]}" + (f" --threads {task[6]}" if task[6] else "")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("groups", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--out", default=str(C.REPO_ROOT / "tasks"))
    a = p.parse_args()
    if a.list or not a.groups:
        for name, fn in GROUPS.items():
            tasks = fn()
            pending = sum(not is_done(t) for t in tasks)
            print(f"{name:16s} pending {pending:4d} / {len(tasks):4d}")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.groups:
        pending = [to_args(t) for t in GROUPS[name]() if not is_done(t)]
        (out / f"{name}.txt").write_text("\n".join(pending) + ("\n" if pending else ""))
        print(f"{name}: {len(pending)} pending tasks -> {out / (name + '.txt')}")


if __name__ == "__main__":
    main()
