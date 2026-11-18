"""Training, prediction and weight-space utilities shared by every stage."""
from __future__ import annotations

import time
from typing import Dict, Optional

import keras
import numpy as np
import tensorflow as tf

from . import config as C
from .data import make_dataset, make_predict_dataset
from .models import apply_policy, find_backbone, preprocess_fn


def callbacks(patience: int, lr_patience: int):
    return [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor=C.MONITOR, patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=C.MONITOR, factor=C.LR_FACTOR,
                                          patience=lr_patience, min_lr=C.MIN_LR),
    ]


def fit(model, train_ds, val_ds, *, lr: float, epochs: int, patience: int, lr_patience: int,
        class_weight: Dict[int, float]) -> Dict:
    """Compiles and fits; early stopping restores the weights of the best epoch."""
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                     callbacks=callbacks(patience, lr_patience), verbose=2)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    vl = h.get(C.MONITOR, [np.nan])
    return {"epochs_run": len(vl), "best_epoch": int(np.nanargmin(vl)) + 1,
            "best_val_loss": float(np.nanmin(vl)), "seconds": time.time() - t0, "history": h}


def datasets(backbone: str, train_x, train_y, val_x, val_y, seed: int):
    pp = preprocess_fn(backbone)
    tr = make_dataset(train_x, train_y, pp, training=True, seed=seed)
    va = make_dataset(val_x, val_y, pp, training=False)
    return tr, va


def train_two_stage(model, backbone: str, train_x, train_y, val_x, val_y, class_weight, seed: int) -> Dict:
    """Source recipe (Table 4): frozen backbone, then top-30 % fine-tuning."""
    tr, va = datasets(backbone, train_x, train_y, val_x, val_y, seed)
    apply_policy(model, "head")
    h1 = fit(model, tr, va, lr=C.HEAD_LR, epochs=C.HEAD_EPOCHS, patience=C.HEAD_PATIENCE,
             lr_patience=C.LR_PATIENCE, class_weight=class_weight)
    apply_policy(model, "partial")
    h2 = fit(model, tr, va, lr=C.FINETUNE_LR, epochs=C.FINETUNE_EPOCHS,
             patience=C.FINETUNE_PATIENCE, lr_patience=C.LR_PATIENCE, class_weight=class_weight)
    return {"stage1": h1, "stage2": h2}


def adapt_strategy(model, strategy: str, backbone: str, train_x, train_y, val_x, val_y,
                   class_weight, seed: int) -> Dict:
    """Target-domain adaptation strategies; the six-way head is kept (no re-init).

    lp    linear probe: only the classifier layer is trained
    ftl   fine-tune last: attention + classifier + top 30 % backbone layers
    fta   fine-tune all: every layer except BatchNorm
    lpft  linear probe, then fine-tune all [Kumar et al., 2022]
    """
    tr, va = datasets(backbone, train_x, train_y, val_x, val_y, seed)
    kw = dict(class_weight=class_weight, lr_patience=C.TL_LR_PATIENCE, patience=C.TL_PATIENCE)
    out = {}
    if strategy in ("lp", "lpft"):
        apply_policy(model, "lp")
        out["lp"] = fit(model, tr, va, lr=C.LP_LR, epochs=C.LP_EPOCHS, **kw)
    if strategy == "ftl":
        apply_policy(model, "partial")
        out["ft"] = fit(model, tr, va, lr=C.FT_LR, epochs=C.FT_EPOCHS, **kw)
    if strategy in ("fta", "lpft"):
        apply_policy(model, "all")
        out["ft"] = fit(model, tr, va, lr=C.FT_LR, epochs=C.FT_EPOCHS, **kw)
    return out


def train_from_imagenet(model, backbone: str, train_x, train_y, val_x, val_y, class_weight,
                        seed: int) -> Dict:
    """Local-only baseline: same LP-FT recipe as `adapt_strategy` but from ImageNet weights."""
    tr, va = datasets(backbone, train_x, train_y, val_x, val_y, seed)
    kw = dict(class_weight=class_weight, lr_patience=C.TL_LR_PATIENCE, patience=C.TL_PATIENCE)
    apply_policy(model, "head")
    h1 = fit(model, tr, va, lr=C.LP_LR, epochs=C.LP_EPOCHS, **kw)
    apply_policy(model, "all")
    h2 = fit(model, tr, va, lr=C.FT_LR, epochs=C.FT_EPOCHS, **kw)
    return {"lp": h1, "ft": h2}


# =============================================================================
# Inference
# =============================================================================

def predict(model, backbone: str, images: np.ndarray, with_embedding: bool = False):
    ds = make_predict_dataset(images, preprocess_fn(backbone))
    if not with_embedding:
        return model.predict(ds, verbose=0).astype(np.float32)
    emb_model = keras.Model(model.input, [model.get_layer("embedding").output, model.output])
    emb, probs = emb_model.predict(ds, verbose=0)
    return probs.astype(np.float32), emb.astype(np.float16)


def clone(model):
    """Copy of ``model`` with identical weights *and* preprocessing state.

    ``keras.layers.Normalization`` (inside the EfficientNet backbones) caches its
    mean/variance as constants when built; ``set_weights`` does not refresh that
    cache, so a plain ``clone_model`` + ``set_weights`` would silently drop the
    ImageNet input normalisation.  ``finalize_state`` rebuilds it.
    """
    m = keras.models.clone_model(model)
    m.set_weights(model.get_weights())
    for layer in m._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, keras.layers.Normalization):
            layer.finalize_state()
    x = np.random.default_rng(0).uniform(0, 255, (2,) + tuple(model.input_shape[1:])).astype("float32")
    if not np.allclose(model(x, training=False), m(x, training=False), atol=1e-2):
        raise RuntimeError("clone() does not reproduce the original model")
    return m


def adapt_batchnorm(model, backbone: str, images: np.ndarray, batch_size: int = 32):
    """AdaBN [Li et al., 2018]: re-estimates BatchNorm statistics on unlabelled target images.

    Moving statistics are replaced by the running average of per-batch statistics
    (momentum k/(k+1) for the k-th batch).  No label and no gradient is used.
    """
    m = clone(model)
    bb = find_backbone(m)
    bns = [l for l in bb.layers if isinstance(l, keras.layers.BatchNormalization)]
    if not bns:
        return m, 0
    old = [(l.trainable, l.momentum) for l in bns]
    bb.trainable = True
    for l in bns:
        l.trainable = True
    ds = make_predict_dataset(images, preprocess_fn(backbone), batch_size=batch_size)
    for k, x in enumerate(ds):
        for l in bns:
            l.momentum = k / (k + 1.0)
        bb(x, training=True)
    for l, (tr, mom) in zip(bns, old):
        l.trainable, l.momentum = tr, mom
    return m, len(bns)


def _bn_input_moments(bb, bns, ds):
    """Per-channel mean and variance of the input of every BatchNorm layer."""
    probe = keras.Model(bb.input, [l.input for l in bns])
    n, s1, s2 = 0.0, None, None
    for x in ds:
        outs = probe(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        f = [tf.cast(o, tf.float32) for o in outs]
        axes = [list(range(len(o.shape) - 1)) for o in f]
        b1 = [tf.reduce_mean(o, axis=a).numpy() for o, a in zip(f, axes)]
        b2 = [tf.reduce_mean(tf.square(o), axis=a).numpy() for o, a in zip(f, axes)]
        w = x.shape[0]
        s1 = [v * w for v in b1] if s1 is None else [a + v * w for a, v in zip(s1, b1)]
        s2 = [v * w for v in b2] if s2 is None else [a + v * w for a, v in zip(s2, b2)]
        n += w
    mean = [a / n for a in s1]
    var = [np.maximum(b / n - m ** 2, 1e-12) for b, m in zip(s2, mean)]
    return mean, var


def align_batchnorm(model, backbone: str, source_images: np.ndarray, target_images: np.ndarray,
                    batch_size: int = 32, depth: float = 1.0):
    """Label-free BatchNorm statistics alignment (AdaBN generalised to frozen-BN training).

    The source models keep ImageNet BatchNorm statistics frozen, so classic AdaBN
    (replacing them by target statistics) would also change the normalisation the
    trained layers were fitted to.  Instead, each BN layer gets statistics
    (m', v') such that target activations are normalised as the source
    activations were: with s = sqrt(v + eps) and the per-channel input moments
    (mu_S, var_S) on unlabelled source images and (mu_T, var_T) on unlabelled
    target images, r = sqrt((var_T + eps) / (var_S + eps)), s' = r s and
    m' = mu_T - r (mu_S - m).  The target moments of a layer depend on every
    earlier (already aligned) layer, so layers are aligned one at a time in
    topological order; updating all layers at once from a single pass corrects
    deep layers for shifts that earlier layers then remove, and diverges.
    ``depth`` < 1 aligns only the first fraction of BN layers.
    """
    m = clone(model)
    bb = find_backbone(m)
    bns = [l for l in bb.layers if isinstance(l, keras.layers.BatchNormalization)]
    bns = bns[:int(round(depth * len(bns)))]
    if not bns:
        return m, 0
    pp = preprocess_fn(backbone)
    ds_s = make_predict_dataset(source_images, pp, batch_size=batch_size)
    ds_t = make_predict_dataset(target_images, pp, batch_size=batch_size)
    mu_s, var_s = _bn_input_moments(bb, bns, ds_s)
    for l, ms, vs in zip(bns, mu_s, var_s):
        (mt,), (vt,) = _bn_input_moments(bb, [l], ds_t)
        mm = l.moving_mean.numpy().astype(np.float64)
        mv = l.moving_variance.numpy().astype(np.float64)
        ratio = np.sqrt((vt + l.epsilon) / (vs + l.epsilon))
        s_new = np.sqrt(mv + l.epsilon) * ratio
        l.moving_variance.assign(np.maximum(s_new ** 2 - l.epsilon, 0.0).astype(np.float32))
        l.moving_mean.assign((mt - ratio * (ms - mm)).astype(np.float32))
    return m, len(bns)


def interpolate_weights(w0, w1, alpha: float):
    """WiSE-FT [Wortsman et al., 2022]: (1 - alpha) * zero-shot + alpha * fine-tuned."""
    return [(1.0 - alpha) * a + alpha * b if np.issubdtype(a.dtype, np.floating) else b
            for a, b in zip(w0, w1)]
