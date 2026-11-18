"""Backbones, attention variants and layer-freezing utilities.

Every model shares the same classification head::

    backbone -> [attention] -> GAP ("embedding") -> Dropout(0.3) -> Dense(6, softmax)

so that baseline and attention variants differ *only* by the attention block.
The main attention variant is a single CBAM [Woo et al., 2018] applied to the
output of the last convolutional stage ("terminal-stage CBAM").  Ablations:
channel-only CBAM, spatial-only CBAM, a Squeeze-and-Excitation block, and a two-stage variant that additionally refines the
output of the penultimate stage.
"""
from __future__ import annotations

from typing import Callable, Tuple

import keras
import numpy as np
import tensorflow as tf

from . import config as C

L = keras.layers

BACKBONE_SPECS = {
    "mobilenetv2": ("MobileNetV2", "mobilenet_v2"),
    "mobilenetv3large": ("MobileNetV3Large", "mobilenet_v3"),
    "efficientnetb0": ("EfficientNetB0", "efficientnet"),
    "densenet121": ("DenseNet121", "densenet"),
    "efficientnetb3": ("EfficientNetB3", "efficientnet"),
    "resnet50": ("ResNet50", "resnet"),
    "inceptionv3": ("InceptionV3", "inception_v3"),
    "vgg16": ("VGG16", "vgg16"),
    "vgg19": ("VGG19", "vgg19"),
    "efficientnetb5": ("EfficientNetB5", "efficientnet"),
    "resnet101": ("ResNet101", "resnet"),
}

PRETTY = {
    "mobilenetv2": "MobileNetV2", "mobilenetv3large": "MobileNetV3Large",
    "efficientnetb0": "EfficientNetB0", "densenet121": "DenseNet121",
    "efficientnetb3": "EfficientNetB3", "resnet50": "ResNet50",
    "inceptionv3": "InceptionV3", "vgg16": "VGG16", "vgg19": "VGG19",
    "efficientnetb5": "EfficientNetB5", "resnet101": "ResNet101",
}
VARIANT_PRETTY = {
    "none": "", "cbam": " + CBAM", "se": " + SE", "cbam_channel": " + CBAM-C",
    "cbam_spatial": " + CBAM-S", "cbam_2stage": " + CBAM-2S",
}


def pretty_name(backbone: str, variant: str = "none") -> str:
    return PRETTY[backbone] + VARIANT_PRETTY[variant]


def preprocess_fn(backbone: str) -> Callable:
    module = getattr(keras.applications, BACKBONE_SPECS[backbone][1])
    return module.preprocess_input


# =============================================================================
# Attention layers
# =============================================================================

@keras.saving.register_keras_serializable(package="peachleaf")
class ChannelPool(L.Layer):
    """Channel-wise average and max pooling -> (H, W, 2) (CBAM Eq. 5)."""

    def call(self, x):
        return tf.concat([tf.reduce_mean(x, axis=-1, keepdims=True),
                          tf.reduce_max(x, axis=-1, keepdims=True)], axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 2)


def channel_attention(x, reduction: int, name: str):
    ch = x.shape[-1]
    hidden = max(1, ch // reduction)
    mlp1 = L.Dense(hidden, activation="relu", name=f"{name}_mlp1")
    mlp2 = L.Dense(ch, name=f"{name}_mlp2")
    avg = mlp2(mlp1(L.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)))
    mx = mlp2(mlp1(L.GlobalMaxPooling2D(keepdims=True, name=f"{name}_gmp")(x)))
    att = L.Activation("sigmoid", name=f"{name}_sigmoid")(L.Add(name=f"{name}_add")([avg, mx]))
    return L.Multiply(name=f"{name}_scale")([x, att])


def spatial_attention(x, name: str, kernel: int = 7):
    pooled = ChannelPool(name=f"{name}_pool")(x)
    att = L.Conv2D(1, kernel, padding="same", activation="sigmoid", use_bias=False,
                   name=f"{name}_conv")(pooled)
    return L.Multiply(name=f"{name}_scale")([x, att])


def cbam(x, reduction: int = C.CBAM_REDUCTION, name: str = "cbam",
         use_channel: bool = True, use_spatial: bool = True):
    if use_channel:
        x = channel_attention(x, reduction, f"{name}_ch")
    if use_spatial:
        x = spatial_attention(x, f"{name}_sp")
    return x


def squeeze_excitation(x, reduction: int = C.CBAM_REDUCTION, name: str = "se"):
    ch = x.shape[-1]
    s = L.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
    s = L.Dense(max(1, ch // reduction), activation="relu", name=f"{name}_fc1")(s)
    s = L.Dense(ch, activation="sigmoid", name=f"{name}_fc2")(s)
    return L.Multiply(name=f"{name}_scale")([x, s])


# =============================================================================
# Model construction
# =============================================================================

def build_backbone(backbone: str, img_size: int = C.IMG_SIZE, weights: str | None = "imagenet"):
    cls_name = BACKBONE_SPECS[backbone][0]
    cls = getattr(keras.applications, cls_name)
    kwargs = dict(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    base = cls(**kwargs)
    return keras.Model(base.input, base.output, name=f"backbone_{backbone}")


def _penultimate_stage_output(base):
    """Output of the penultimate stage.

    It is the last merge layer (Add/Concatenate) at the smallest resolution
    larger than the final feature map; for plain networks (VGG) it falls back
    to the last layer at that resolution.
    """
    final_hw = base.output.shape[1]
    layers = []
    for layer in base.layers:
        try:
            shp = layer.output.shape
        except (AttributeError, ValueError):
            continue
        if len(shp) == 4 and shp[1] is not None and shp[1] > final_hw \
                and not isinstance(layer, L.ZeroPadding2D):
            layers.append((shp[1], layer))
    if not layers:
        raise ValueError(f"No penultimate stage found for {base.name}")
    r_prev = min(r for r, _ in layers)
    at_res = [l for r, l in layers if r == r_prev]
    merges = [l for l in at_res if isinstance(l, (L.Add, L.Concatenate))]
    return (merges or at_res)[-1].output


def build_model(backbone: str, variant: str = "none", num_classes: int = C.NUM_CLASSES,
                img_size: int = C.IMG_SIZE, weights: str | None = "imagenet") -> keras.Model:
    if variant not in C.ATTENTION_VARIANTS:
        raise ValueError(variant)
    base = build_backbone(backbone, img_size, weights)
    base.trainable = False
    inputs = keras.Input((img_size, img_size, 3), name="image")

    if variant == "cbam_2stage":
        feat = keras.Model(base.input, [_penultimate_stage_output(base), base.output],
                           name=f"backbone_{backbone}")
        mid, x = feat(inputs, training=False)
        mid = cbam(mid, name="attn_mid")
        x = cbam(x, name="attn")
        x = L.Concatenate(name="embedding")([
            L.GlobalAveragePooling2D(name="gap_mid")(mid),
            L.GlobalAveragePooling2D(name="gap")(x)])
    else:
        # training=False keeps BatchNorm in inference mode even when unfrozen
        x = base(inputs, training=False)
        if variant == "cbam":
            x = cbam(x, name="attn")
        elif variant == "cbam_channel":
            x = cbam(x, name="attn", use_spatial=False)
        elif variant == "cbam_spatial":
            x = cbam(x, name="attn", use_channel=False)
        elif variant == "se":
            x = squeeze_excitation(x, name="attn")
        x = L.GlobalAveragePooling2D(name="embedding")(x)

    x = L.Dropout(C.DROPOUT, name="dropout")(x)
    # float32 output for numerically stable softmax under mixed precision
    outputs = L.Dense(num_classes, activation="softmax", dtype="float32", name="classifier")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_{variant}")


def load_model(path) -> keras.Model:
    return keras.models.load_model(path, compile=False)


def find_backbone(model: keras.Model) -> keras.Model:
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No backbone sub-model found")


def embedding_model(model: keras.Model) -> keras.Model:
    return keras.Model(model.input, [model.get_layer("embedding").output, model.output])


# =============================================================================
# Freezing policies
# =============================================================================

def _is_bn(layer) -> bool:
    return isinstance(layer, L.BatchNormalization)


def set_backbone_trainable(backbone: keras.Model, fraction: float) -> None:
    """Unfreezes the top ``fraction`` of backbone layers; BatchNorm stays frozen."""
    backbone.trainable = True
    n = len(backbone.layers)
    start = int(round(n * (1.0 - fraction)))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= start) and not _is_bn(layer)


def head_layers(model: keras.Model):
    return [l for l in model.layers if not isinstance(l, keras.Model)]


def apply_policy(model: keras.Model, policy: str) -> None:
    """Freezing policies shared by source training and target adaptation.

    head      backbone frozen; attention + classifier trainable (source stage 1)
    partial   + top 30 % backbone layers (source stage 2, target "FTL")
    lp        classifier only (linear probe on frozen source features)
    all       every layer except BatchNorm statistics/affine (target "FTA")
    """
    backbone = find_backbone(model)
    if policy == "lp":
        backbone.trainable = False
        for l in head_layers(model):
            l.trainable = l.name == "classifier"
    elif policy == "head":
        backbone.trainable = False
        for l in head_layers(model):
            l.trainable = True
    elif policy == "partial":
        set_backbone_trainable(backbone, C.UNFREEZE_FRACTION)
        for l in head_layers(model):
            l.trainable = True
    elif policy == "all":
        set_backbone_trainable(backbone, 1.0)
        for l in head_layers(model):
            l.trainable = True
    else:
        raise ValueError(policy)


def _n(w) -> int:
    return int(np.prod(w.shape))


def count_params(model: keras.Model) -> Tuple[int, int]:
    total = sum(_n(w) for w in model.weights)
    attn = sum(_n(w) for l in head_layers(model) if l.name.startswith("attn") for w in l.weights)
    return total, attn
