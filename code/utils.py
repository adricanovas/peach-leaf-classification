from typing import Tuple
import cv2
import albumentations as A
import numpy as np
import tensorflow as tf
from pathlib import Path


def create_augmentation_pipeline(is_training: bool = True, img_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    if is_training:
        return A.Compose([
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.RandomResizedCrop(
                size=(img_size[0], img_size[1]),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
                p=0.3
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),

            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),

            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=4, p=0.2),

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.3
            ),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
        ])

def load_paths_and_labels(data_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths, labels = [], []
    for cdir in class_dirs:
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))
                labels.append(class_to_idx[cdir.name])

    return np.array(paths), np.array(labels), class_names


def make_dataset(paths, labels, img_size, preprocess_fn, batch_size=32, training=False):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    num_classes = tf.reduce_max(labels) + 1

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)
        img = preprocess_fn(img)
        y = tf.one_hot(y, depth=num_classes)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(model_name, num_classes, img_size, model_specs):
    base_cls, _, _ = model_specs[model_name]
    base = base_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model