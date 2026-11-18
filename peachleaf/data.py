"""Dataset manifest, grouped splits and tf.data input pipelines.

Leaves are cropped from multi-leaf photographs, so every split in this project
is *grouped by parent photograph*: crops of the same photo never
appear on both sides of a train/validation/test boundary.  The parent photo is
recovered from the crop file name ``<photo>_ann<k>.<ext>`` written by the CVAT
export.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import config as C

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_ANN_SUFFIX = re.compile(r"_ann\d+$")


# =============================================================================
# Manifest
# =============================================================================

def parent_photo(file_name: str) -> str:
    return _ANN_SUFFIX.sub("", Path(file_name).stem)


def _local_campaign(group: str) -> str:
    # Multi-leaf photographs of the first campaign were exported with the
    # "open_field-local" prefix; single-leaf photographs of the second
    # campaign use "local_field-<class>".
    return "2024" if group.startswith("open_field-local") else "2025"


_PHONE_SIZES = {(1920, 2560), (2560, 1920), (3456, 4608), (4608, 3456)}


@lru_cache(maxsize=1)
def _public_image_sizes() -> Dict[str, tuple]:
    path = C.DATA_ROOT / "public_dataset" / "anotaciones_def_public.json"
    with path.open() as f:
        images = json.load(f)["images"]
    return {Path(i["file_name"]).stem: (i["width"], i["height"]) for i in images}


def _public_source(group: str) -> str:
    """Source of a public photograph, recovered from its resolution.

    File names were normalised before annotation, so the source is inferred from
    the image size stored in the COCO export: the Mendeley images [Farman et al.]
    were taken with smartphones at 1920x2560 or 3456x4608 pixels, whereas the
    PlantDoc images are web images of heterogeneous size (25 of the 26 match the
    size of a PlantDoc peach-leaf image exactly).  The two "shot hole" photographs,
    absent from the final COCO export, are 1920x2560 Mendeley images.
    """
    size = _public_image_sizes().get(group)
    if size is None or size in _PHONE_SIZES:
        return "mendeley"
    return "plantdoc"


def build_manifest() -> pd.DataFrame:
    rows = []
    for domain, root in (("public", C.PUBLIC_DIR), ("local", C.LOCAL_DIR)):
        for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
            if cdir.name not in C.CLASS_TO_IDX:
                raise ValueError(f"Unknown class folder {cdir}")
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                group = parent_photo(f.name)
                rows.append({
                    "relpath": str(f.relative_to(C.DATA_ROOT)),
                    "domain": domain,
                    "label": cdir.name,
                    "label_idx": C.CLASS_TO_IDX[cdir.name],
                    # groups are namespaced by domain so that ids never collide
                    "group": f"{domain}:{group}",
                    "campaign": _local_campaign(group) if domain == "local" else "public",
                    "source": _public_source(group) if domain == "public" else "orchard",
                })
    df = pd.DataFrame(rows)
    return df


def load_manifest() -> pd.DataFrame:
    if not C.MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"{C.MANIFEST_CSV} not found; run `python -m peachleaf.splits` first.")
    return pd.read_csv(C.MANIFEST_CSV)


@lru_cache(maxsize=1)
def load_splits() -> Dict:
    with (C.SPLITS_DIR / "splits.json").open() as f:
        return json.load(f)


def select(manifest: pd.DataFrame, relpaths: Sequence[str]) -> pd.DataFrame:
    idx = manifest.set_index("relpath")
    return idx.loc[list(relpaths)].reset_index()


# =============================================================================
# Image loading (letterbox, cached in memory)
# =============================================================================

def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Pad to a square with black pixels and resize (no aspect distortion).

    Crops are polygon-masked leaves on a black background, so zero padding
    extends the existing background instead of stretching elongated leaves.  INTER_AREA is used for down-sampling, INTER_CUBIC otherwise.
    """
    import cv2

    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = img
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)


def read_rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_images(relpaths: Sequence[str], size: int = C.IMG_SIZE) -> np.ndarray:
    out = np.empty((len(relpaths), size, size, 3), dtype=np.uint8)
    for i, rp in enumerate(relpaths):
        out[i] = letterbox(read_rgb(C.DATA_ROOT / rp), size)
    return out


class ImageBank:
    """Lazily decodes every image once per process and serves arrays by relpath."""

    def __init__(self, size: int = C.IMG_SIZE):
        self.size = size
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, relpaths: Sequence[str]) -> np.ndarray:
        missing = [rp for rp in relpaths if rp not in self._cache]
        if missing:
            imgs = load_images(missing, self.size)
            self._cache.update(zip(missing, imgs))
        return np.stack([self._cache[rp] for rp in relpaths])


# =============================================================================
# Augmentation and tf.data
# =============================================================================

def augmentation_pipeline(size: int = C.IMG_SIZE):
    """On-the-fly augmentation (Section 2.3); identical for every model."""
    import albumentations as A
    import cv2

    return A.Compose([
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
        A.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), ratio=(0.9, 1.1),
                            interpolation=cv2.INTER_LINEAR, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=4, p=0.2),
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 scale=(0.95, 1.05), rotate=0, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3),
    ], seed=None)


def make_dataset(images: np.ndarray, labels: np.ndarray, preprocess_fn, *,
                 training: bool, augment: bool = True, batch_size: int = C.BATCH_SIZE,
                 seed: int = 0, num_classes: int = C.NUM_CLASSES):
    """Builds a tf.data pipeline from in-memory uint8 images.

    ``num_classes`` is explicit: the one-hot depth must not depend on which
    classes happen to be present in a split (bug in the previous version).
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((images, labels.astype(np.int32)))
    if training:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
        if augment:
            aug = augmentation_pipeline(images.shape[1])
            rng = np.random.default_rng(seed)
            aug.set_random_seed(int(rng.integers(2**31 - 1)))

            def _np_aug(x):
                return aug(image=x)["image"].astype(np.uint8)

            def _tf_aug(x, y):
                x = tf.numpy_function(_np_aug, [x], tf.uint8)
                x.set_shape(images.shape[1:])
                return x, y

            ds = ds.map(_tf_aug, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    def _prep(x, y):
        x = preprocess_fn(tf.cast(x, tf.float32))
        return x, tf.one_hot(y, num_classes)

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_predict_dataset(images: np.ndarray, preprocess_fn, batch_size: int = 64):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(lambda x: preprocess_fn(tf.cast(x, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(labels: np.ndarray, num_classes: int = C.NUM_CLASSES) -> Dict[int, float]:
    """Balanced inverse-frequency weights over the classes present in ``labels``.

    Absent classes get weight 0 (they contribute no samples anyway).  This is the
    only imbalance correction used: there is no oversampling.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    present = counts > 0
    w = np.zeros(num_classes)
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    return {i: float(v) for i, v in enumerate(w)}


def grouped_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draws roughly ``n`` leaves by whole photographs, keeping >=1 photo per class.

    Used for the learning curve: whole photos are drawn so that the
    budget reflects real annotation effort and no photo is split.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby("group")
    chosen: List[str] = []
    # guarantee coverage of every class present in the pool
    for label in sorted(df["label"].unique()):
        cand = df.loc[df["label"] == label, "group"].unique()
        g = rng.choice(cand)
        if g not in chosen:
            chosen.append(g)
    order = rng.permutation(df["group"].unique())
    for g in order:
        if sum(len(groups.get_group(c)) for c in chosen) >= n:
            break
        if g not in chosen:
            chosen.append(g)
    return df[df["group"].isin(chosen)].reset_index(drop=True)
