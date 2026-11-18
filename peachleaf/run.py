"""Entry point for one experimental unit (one process per unit).

Each unit writes ``runs/<stage>/<unit>/{preds.npz, meta.json, DONE}``.
``preds.npz`` stores per-leaf six-way probabilities for every evaluated
partition, so all metrics, confidence intervals and paired tests are computed
afterwards by ``peachleaf.analysis`` without retraining.

Examples::

    python -m peachleaf.run source --backbone efficientnetb0 --variant cbam --repeat 1 --fold 3
    python -m peachleaf.run adapt  --backbone efficientnetb0 --variant cbam --repeat 2 --fold 3
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import argparse
import json
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C


# =============================================================================
# Unit bookkeeping
# =============================================================================

def mkdirs(path: Path, attempts: int = 8) -> None:
    """``mkdir -p`` that tolerates concurrent creation on network file systems.

    When several processes create the same parent at once, ``os.mkdir``
    raises EEXIST but the client's stale attribute cache can still report the
    directory as missing, so ``exist_ok=True`` re-raises ``FileExistsError``.
    Retrying after a short pause lets the cache catch up.
    """
    for i in range(attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except (FileExistsError, FileNotFoundError):
            if i == attempts - 1:
                raise
            time.sleep(0.5 * (i + 1))


def unit_dir(stage: str, backbone: str, variant: str, repeat: int, fold: int) -> Path:
    return C.RUNS_ROOT / stage / f"{backbone}__{variant}" / f"r{repeat}_f{fold}"


def source_model_path(backbone: str, variant: str, fold: int) -> Path:
    # target-domain stages always start from the repeat-1 source model of the
    # same fold index: each local fold k is paired with a different source model
    return unit_dir("source", backbone, variant, 1, fold) / "model.keras"


class Unit:
    def __init__(self, out: Path, args: argparse.Namespace):
        self.out, self.args = out, args
        self.tmp = out.with_name(out.name + ".tmp")
        self.preds = {}
        self.meta = {"args": vars(args), "host": socket.gethostname(),
                     "python": platform.python_version(), "started": time.time()}

    def done(self) -> bool:
        return (self.out / "DONE").exists()

    def add(self, key: str, probs: np.ndarray, relpaths, emb: np.ndarray | None = None):
        self.preds[f"{key}|probs"] = probs.astype(np.float16)
        self.preds[f"{key}|relpath"] = np.asarray(relpaths)
        if emb is not None:
            self.preds[f"{key}|emb"] = emb

    def commit(self, model=None):
        import tensorflow as tf
        import keras

        shutil.rmtree(self.tmp, ignore_errors=True)
        mkdirs(self.tmp)
        np.savez_compressed(self.tmp / "preds.npz", **self.preds)
        self.meta.update(finished=time.time(), tf=tf.__version__, keras=keras.__version__,
                         gpu=[d.name for d in tf.config.list_physical_devices("GPU")])
        with (self.tmp / "meta.json").open("w") as f:
            json.dump(self.meta, f, indent=1, default=float)
        if model is not None:
            model.save(self.tmp / "model.keras")
        (self.tmp / "DONE").touch()
        shutil.rmtree(self.out, ignore_errors=True)
        self.tmp.rename(self.out)


def setup_runtime(seed: int):
    import keras
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if C.MIXED_PRECISION and tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
    keras.utils.set_random_seed(seed)


def xy(manifest, bank, relpaths):
    lab = manifest.set_index("relpath").loc[list(relpaths), "label_idx"].to_numpy()
    return bank.get(list(relpaths)), lab.astype(np.int32)


# =============================================================================
# Stages
# =============================================================================

def stage_source(a):
    """Public-domain CV unit: train on 4/5 of the development set, early-stop on the 5th.

    Every fold model is also scored on the fixed public hold-out test and on all
    182 orchard leaves (true zero-shot, no target label is ever used).
    AdaBN predictions re-estimate BatchNorm statistics on the unlabelled
    orchard images (transductive, label-free baseline).
    """
    from .data import ImageBank, class_weights, load_manifest, load_splits
    from .models import build_model
    from .train import predict, train_two_stage

    unit = Unit(unit_dir("source", a.backbone, a.variant, a.repeat, a.fold), a)
    if unit.done() and not a.force:
        return print("already done:", unit.out)
    seed = C.seed_for(a.repeat, a.fold)
    setup_runtime(seed)
    man, sp, bank = load_manifest(), load_splits(), ImageBank()
    f = sp["public_cv"][str(a.repeat)][a.fold - 1]
    trx, try_ = xy(man, bank, f["train"])
    vax, vay = xy(man, bank, f["val"])
    model = build_model(a.backbone, a.variant)
    unit.meta["train"] = train_two_stage(model, a.backbone, trx, try_, vax, vay,
                                         class_weights(try_), seed)
    keep = a.repeat in C.SAVE_MODEL_REPEATS
    unit.add("public_val", predict(model, a.backbone, vax), f["val"])
    for key in ("public_test", "local_all"):
        imgs, _ = xy(man, bank, sp[key])
        if keep:
            probs, emb = predict(model, a.backbone, imgs, with_embedding=True)
            unit.add(key, probs, sp[key], emb)
        else:
            unit.add(key, predict(model, a.backbone, imgs), sp[key])
    unit.commit(model if keep else None)


def stage_adabn(a):
    """Label-free BatchNorm statistics alignment of the repeat-1 source models."""
    from .data import ImageBank, load_manifest, load_splits
    from .models import load_model
    from .train import adapt_batchnorm, align_batchnorm, predict

    unit = Unit(unit_dir("adabn", a.backbone, a.variant, 1, a.fold), a)
    if unit.done() and not a.force:
        return print("already done:", unit.out)
    setup_runtime(C.seed_for(1, a.fold, offset=1300))
    man, sp, bank = load_manifest(), load_splits(), ImageBank()
    src = load_model(source_model_path(a.backbone, a.variant, a.fold))
    srcx, _ = xy(man, bank, sp["public_cv"]["1"][a.fold - 1]["train"])
    locx, _ = xy(man, bank, sp["local_all"])
    pubx, _ = xy(man, bank, sp["public_test"])
    for method, fn in (("adabn", lambda: adapt_batchnorm(src, a.backbone, locx)),
                       ("align", lambda: align_batchnorm(src, a.backbone, srcx, locx))):
        model, n_bn = fn()
        unit.meta[f"{method}_bn_layers"] = n_bn
        if not n_bn:  # VGG: no BatchNorm, nothing to adapt
            continue
        unit.add(f"{method}/local_all", predict(model, a.backbone, locx), sp["local_all"])
        unit.add(f"{method}/public_test", predict(model, a.backbone, pubx), sp["public_test"])
        del model
    unit.commit()


def _local_fold(sp, repeat, fold):
    return sp["local_cv"][str(repeat)][fold - 1]


def stage_adapt(a):
    """Target-domain adaptation from the source model.

    For each strategy the same source checkpoint is fine-tuned on the local
    training part, early-stopped on the grouped inner validation part and
    scored on (i) the untouched local test fold and (ii) the public hold-out
    test, so forgetting of the source domain is measured directly.
    WiSE-FT weight interpolation is evaluated for the full fine-tuning runs.
    """
    from .data import ImageBank, class_weights, load_manifest, load_splits
    from .models import load_model
    from .train import adapt_strategy, clone, interpolate_weights, predict

    unit = Unit(unit_dir("adapt", a.backbone, a.variant, a.repeat, a.fold), a)
    if unit.done() and not a.force:
        return print("already done:", unit.out)
    src = source_model_path(a.backbone, a.variant, a.fold)
    if not (src.parent / "DONE").exists():
        sys.exit(f"missing source model {src}")
    seed = C.seed_for(a.repeat, a.fold, offset=500)
    setup_runtime(seed)
    man, sp, bank = load_manifest(), load_splits(), ImageBank()
    f = _local_fold(sp, a.repeat, a.fold)
    trx, try_ = xy(man, bank, f["train"])
    vax, vay = xy(man, bank, f["val"])
    tex, _ = xy(man, bank, f["test"])
    pubx, _ = xy(man, bank, sp["public_test"])
    evals = {"local_val": (vax, f["val"]), "local_test": (tex, f["test"]),
             "public_test": (pubx, sp["public_test"])}
    source = load_model(src)
    w0 = source.get_weights()
    for k, (x, rel) in evals.items():
        unit.add(f"zs/{k}", predict(source, a.backbone, x), rel)
    cw = class_weights(try_)
    unit.meta["train"] = {}
    for strategy in a.strategies:
        import keras

        keras.utils.set_random_seed(seed)
        model = clone(source)
        unit.meta["train"][strategy] = adapt_strategy(model, strategy, a.backbone, trx, try_,
                                                      vax, vay, cw, seed)
        for k, (x, rel) in evals.items():
            unit.add(f"{strategy}/{k}", predict(model, a.backbone, x), rel)
        if strategy in ("fta", "lpft"):
            w1 = model.get_weights()
            for alpha in C.WISE_ALPHAS[1:-1]:
                model.set_weights(interpolate_weights(w0, w1, alpha))
                for k, (x, rel) in evals.items():
                    unit.add(f"wise_{strategy}_{alpha:.2f}/{k}", predict(model, a.backbone, x), rel)
        del model
    unit.commit()


def stage_scratch(a):
    """Local-only baseline from ImageNet weights (no public data)."""
    from .data import ImageBank, class_weights, load_manifest, load_splits
    from .models import build_model
    from .train import predict, train_from_imagenet

    unit = Unit(unit_dir("scratch", a.backbone, a.variant, a.repeat, a.fold), a)
    if unit.done() and not a.force:
        return print("already done:", unit.out)
    seed = C.seed_for(a.repeat, a.fold, offset=700)
    setup_runtime(seed)
    man, sp, bank = load_manifest(), load_splits(), ImageBank()
    f = _local_fold(sp, a.repeat, a.fold)
    trx, try_ = xy(man, bank, f["train"])
    vax, vay = xy(man, bank, f["val"])
    tex, _ = xy(man, bank, f["test"])
    model = build_model(a.backbone, a.variant)
    unit.meta["train"] = train_from_imagenet(model, a.backbone, trx, try_, vax, vay,
                                             class_weights(try_), seed)
    unit.add("scratch/local_val", predict(model, a.backbone, vax), f["val"])
    unit.add("scratch/local_test", predict(model, a.backbone, tex), f["test"])
    unit.commit()


def stage_joint(a):
    """Joint public + local training from ImageNet weights.

    Uses the public CV split of repeat 1 / fold k plus the local split (r, k);
    the validation set is the union of both validation parts.
    """
    from .data import ImageBank, class_weights, load_manifest, load_splits
    from .models import build_model
    from .train import predict, train_two_stage

    unit = Unit(unit_dir("joint", a.backbone, a.variant, a.repeat, a.fold), a)
    if unit.done() and not a.force:
        return print("already done:", unit.out)
    seed = C.seed_for(a.repeat, a.fold, offset=900)
    setup_runtime(seed)
    man, sp, bank = load_manifest(), load_splits(), ImageBank()
    pf = sp["public_cv"]["1"][a.fold - 1]
    lf = _local_fold(sp, a.repeat, a.fold)
    trx, try_ = xy(man, bank, pf["train"] + lf["train"])
    vax, vay = xy(man, bank, pf["val"] + lf["val"])
    tex, _ = xy(man, bank, lf["test"])
    pubx, _ = xy(man, bank, sp["public_test"])
    lvx, _ = xy(man, bank, lf["val"])
    model = build_model(a.backbone, a.variant)
    unit.meta["train"] = train_two_stage(model, a.backbone, trx, try_, vax, vay,
                                         class_weights(try_), seed)
    unit.add("joint/local_val", predict(model, a.backbone, lvx), lf["val"])
    unit.add("joint/local_test", predict(model, a.backbone, tex), lf["test"])
    unit.add("joint/public_test", predict(model, a.backbone, pubx), sp["public_test"])
    unit.commit()


def stage_curve(a):
    """Learning curve vs number of labelled orchard leaves.

    Training subsets are drawn by whole photographs from the local training
    part; validation and test parts are unchanged.  Public-pretrained FTA and
    LP-FT are compared with the ImageNet-only LP-FT baseline.
    """
    import keras

    from .data import ImageBank, class_weights, grouped_subsample, load_manifest, load_splits
    from .models import build_model, load_model
    from .train import adapt_strategy, clone, predict, train_from_imagenet

    unit = Unit(unit_dir("curve", a.backbone, a.variant, a.repeat, a.fold), a)
    if unit.done() and not a.force:
        return print("already done:", unit.out)
    src = source_model_path(a.backbone, a.variant, a.fold)
    seed = C.seed_for(a.repeat, a.fold, offset=1100)
    setup_runtime(seed)
    man, sp, bank = load_manifest(), load_splits(), ImageBank()
    f = _local_fold(sp, a.repeat, a.fold)
    pool = man.set_index("relpath").loc[f["train"]].reset_index()
    vax, vay = xy(man, bank, f["val"])
    tex, _ = xy(man, bank, f["test"])
    source = load_model(src)
    unit.meta["subsets"], unit.meta["train"] = {}, {}
    for n in C.LEARNING_CURVE_SIZES + ["all"]:
        sub = pool if n == "all" else grouped_subsample(pool, int(n), seed + (0 if n == "all" else int(n)))
        rel = sub["relpath"].tolist()
        trx, try_ = xy(man, bank, rel)
        cw = class_weights(try_)
        unit.meta["subsets"][str(n)] = {"n_leaves": len(rel), "n_photos": int(sub["group"].nunique())}
        for strategy in ("fta", "lpft"):
            keras.utils.set_random_seed(seed)
            model = clone(source)
            unit.meta["train"][f"{strategy}_{n}"] = adapt_strategy(
                model, strategy, a.backbone, trx, try_, vax, vay, cw, seed)
            unit.add(f"{strategy}_n{n}/local_test", predict(model, a.backbone, tex), f["test"])
            del model
        keras.utils.set_random_seed(seed)
        model = build_model(a.backbone, a.variant)
        unit.meta["train"][f"scratch_{n}"] = train_from_imagenet(model, a.backbone, trx, try_,
                                                                 vax, vay, cw, seed)
        unit.add(f"scratch_n{n}/local_test", predict(model, a.backbone, tex), f["test"])
        del model
    unit.commit()


STAGES = {"source": stage_source, "adabn": stage_adabn, "adapt": stage_adapt, "scratch": stage_scratch,
          "joint": stage_joint, "curve": stage_curve}


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=sorted(STAGES) + ["cost"])
    p.add_argument("--backbone", required=True, choices=C.BACKBONES)
    p.add_argument("--variant", default="none", choices=C.ATTENTION_VARIANTS)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--strategies", nargs="+", default=C.TL_STRATEGIES)
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    sys.stdout.reconfigure(line_buffering=True)
    if a.stage == "cost":
        from .cost import run_cost
        return run_cost(a)
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
