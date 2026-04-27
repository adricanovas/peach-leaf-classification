from pathlib import Path
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils import SpatialPooling, load_paths_and_labels, make_dataset, plot_and_save_confusion_matrix

# =========================
# Config
# =========================
TARGET_DATA_DIR = Path("../data/merged_dataset/local_data")
MODELS = [
    "mobilenetv2",
    "efficientnetb0",
    "mobilenetv3large",
    "densenet121",
    "efficientnetb3",
    "resnet50",
    "inceptionv3",
    "resnet101",
    "vgg16",
    "vgg19",
    "efficientnetb5",
]
IMAGE_SIZES = [192, 224]
STRATEGIES = ["feature_extractor", "fine_tune_last", "fine_tune_all"]

TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 16
BASE_SEED = 5681

TL_EPOCHS = 15
TL_LR = 1e-4
# Linear probe settings to estimate 'before' when class counts differ
PROBE_EPOCHS = 5
PROBE_LR = 1e-3
LAST_N_LAYERS = 20
FREEZE_BATCHNORM = True
USE_AUGMENTATION = True

# Try to pick up CBAM-trained models from a CBAM-specific artifacts dir,
# but fall back to a shared 'artifacts' directory so base models placed in
# artifacts/final_model are picked up as requested.
CBAM_ARTIFACTS_DIR = Path(os.environ.get("CBAM_ARTIFACTS_DIR", "artifacts_cbam"))
CBAM_FINAL_MODEL_DIR = CBAM_ARTIFACTS_DIR / "final_model"
# General artifacts fallback (where base models are expected to live)
GENERAL_ARTIFACTS_FINAL = Path(os.environ.get("ARTIFACTS_DIR", "artifacts")) / "final_model"

CBAM_TL_ARTIFACTS_DIR = Path(os.environ.get("CBAM_TL_ARTIFACTS_DIR", "artifacts_cbam_tl"))
CBAM_TL_CHECKPOINTS_DIR = CBAM_TL_ARTIFACTS_DIR / "checkpoints"
CBAM_TL_FINAL_MODEL_DIR = CBAM_TL_ARTIFACTS_DIR / "final_model"
CBAM_TL_RESULTS_DIR = CBAM_TL_ARTIFACTS_DIR / "results"
RESULTS_PATH = CBAM_TL_RESULTS_DIR / "domain_shift_cbam_tl_results.csv"
BEST_RESULTS_PATH = CBAM_TL_RESULTS_DIR / "domain_shift_cbam_tl_best_by_model.csv"
CBAM_TL_CONFUSION_MATRIX_DIR = CBAM_TL_ARTIFACTS_DIR / "confusionmatrix"

MODEL_SPECS = {
    "mobilenetv2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "efficientnetb0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
    "mobilenetv3large": (
        tf.keras.applications.MobileNetV3Large,
        tf.keras.applications.mobilenet_v3.preprocess_input,
    ),
    "densenet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
    ),
    "efficientnetb3": (
        tf.keras.applications.EfficientNetB3,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
    "resnet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet.preprocess_input,
    ),
    "inceptionv3": (
        tf.keras.applications.InceptionV3,
        tf.keras.applications.inception_v3.preprocess_input,
    ),
    "resnet101": (
        tf.keras.applications.ResNet101,
        tf.keras.applications.resnet.preprocess_input,
    ),
    "vgg16": (
        tf.keras.applications.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
    ),
    "vgg19": (
        tf.keras.applications.VGG19,
        tf.keras.applications.vgg19.preprocess_input,
    ),
    "efficientnetb5": (
        tf.keras.applications.EfficientNetB5,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
}


def _find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No backbone submodel found in loaded CBAM model.")


def _mark_head_trainable(model, trainable):
    for layer in model.layers:
        if not isinstance(layer, tf.keras.Model):
            layer.trainable = trainable


def _custom_objects():
    return {"SpatialPooling": SpatialPooling}


def apply_fine_tuning_strategy(model, strategy):
    backbone = _find_backbone(model)

    if strategy == "feature_extractor":
        backbone.trainable = False
        _mark_head_trainable(model, True)
        return model

    if strategy == "fine_tune_last":
        backbone.trainable = True
        if len(backbone.layers) <= LAST_N_LAYERS:
            cutoff = 0
        else:
            cutoff = len(backbone.layers) - LAST_N_LAYERS
        for layer in backbone.layers[:cutoff]:
            layer.trainable = False
        for layer in backbone.layers[cutoff:]:
            if FREEZE_BATCHNORM and isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        _mark_head_trainable(model, True)
        return model

    if strategy == "fine_tune_all":
        backbone.trainable = True
        if FREEZE_BATCHNORM:
            for layer in backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
        _mark_head_trainable(model, True)
        return model

    raise ValueError(f"Unknown strategy: {strategy}")


def get_class_weight(labels, num_classes):
    cls_w = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    return {i: float(w) for i, w in enumerate(cls_w)}


def evaluate_predictions(model, dataset, y_true):
    y_prob = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def train_tl(model, train_ds, val_ds, class_weight, checkpoint_path):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(TL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TL_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )


def _rebuild_model_for_target_classes(model, num_classes):
    if len(model.layers) >= 2 and isinstance(model.layers[-1], tf.keras.layers.Dense):
        # Reuse backbone + CBAM learned features, replace only the classifier head.
        inputs = model.input
        x = model.layers[-2].output
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="tl_head")(x)
        return tf.keras.Model(inputs, outputs)

    backbone = _find_backbone(model)
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="tl_head")(x)
    return tf.keras.Model(inputs, outputs)


def _load_cbam_base_model(model_name, img_size):
    # Prefer CBAM-specific final model location
    candidates = [
        CBAM_FINAL_MODEL_DIR / f"{model_name}_cbam_img{img_size}_final.keras",
        CBAM_FINAL_MODEL_DIR / f"{model_name}_cbam_best_final.keras",
    ]

    # Also look in the shared artifacts/final_model location (non-CBAM naming)
    candidates += [
        GENERAL_ARTIFACTS_FINAL / f"{model_name}_cbam_img{img_size}_final.keras",
        GENERAL_ARTIFACTS_FINAL / f"{model_name}_cbam_best_final.keras",
        GENERAL_ARTIFACTS_FINAL / f"{model_name}_img{img_size}_final.keras",
        GENERAL_ARTIFACTS_FINAL / f"{model_name}_best_final.keras",
    ]

    for path in candidates:
        if path.exists():
            try:
                # Try loading with CBAM custom objects first (safe if model has SpatialPooling)
                model = tf.keras.models.load_model(path, compile=False, custom_objects=_custom_objects())
            except Exception:
                # Fallback to loading without custom objects (works for regular models)
                model = tf.keras.models.load_model(path, compile=False)
            return model, path

    # Not found
    return None, candidates[0]


def _upsert_result(rows, row):
    filtered = [
        item
        for item in rows
        if not (
            item["model"] == row["model"]
            and int(item["img_size"]) == int(row["img_size"])
            and item["strategy"] == row["strategy"]
        )
    ]
    filtered.append(row)
    return filtered


def main():
    X_target, y_target, class_names = load_paths_and_labels(TARGET_DATA_DIR)
    num_classes = len(class_names)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_target,
        y_target,
        test_size=TEST_SIZE,
        stratify=y_target,
        random_state=BASE_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        stratify=y_trainval,
        random_state=BASE_SEED,
    )

    print(f"Target domain classes: {class_names}")
    print(f"Target samples total: {len(X_target)}")
    print(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    CBAM_TL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    CBAM_TL_FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CBAM_TL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CBAM_TL_CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH).to_dict(orient="records")

    for model_name in MODELS:
        model_name_with_tag = f"{model_name}_cbam"
        _, preprocess_fn = MODEL_SPECS[model_name]
        for img_size in IMAGE_SIZES:
            print(f"\n===== {model_name_with_tag} | img_size={img_size} =====")

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(BASE_SEED)

            base_model, base_model_path = _load_cbam_base_model(model_name, img_size)
            if base_model is None:
                print(f"Skipping (CBAM base model not found): {base_model_path}")
                continue

            source_head_units = int(base_model.output_shape[-1])
            is_same_num_classes = source_head_units == num_classes

            train_ds = make_dataset(
                X_train,
                y_train,
                img_size,
                preprocess_fn,
                BATCH_SIZE,
                training=True,
                use_augmentation=USE_AUGMENTATION,
            )
            val_ds = make_dataset(
                X_val,
                y_val,
                img_size,
                preprocess_fn,
                BATCH_SIZE,
                training=False,
            )
            test_ds = make_dataset(
                X_test,
                y_test,
                img_size,
                preprocess_fn,
                BATCH_SIZE,
                training=False,
            )

            if is_same_num_classes:
                before_metrics = evaluate_predictions(base_model, test_ds, y_test)
                print(
                    "Before TL (target test) -> "
                    f"F1-macro: {before_metrics['f1_macro']:.4f} | "
                    f"F1-weighted: {before_metrics['f1_weighted']:.4f} | "
                    f"Acc: {before_metrics['accuracy']:.4f}"
                )
            else:
                print(
                    f"Class mismatch for {model_name_with_tag}|img{img_size}: "
                    f"source={source_head_units}, target={num_classes}. "
                    "Creating linear probe (train head only) to estimate 'before' performance."
                )
                # Create probe model from base backbone, train only the new head for a few epochs
                probe_base = tf.keras.models.clone_model(base_model)
                probe_base.set_weights(base_model.get_weights())
                probe_model = _rebuild_model_for_target_classes(probe_base, num_classes)
                try:
                    backbone = _find_backbone(probe_model)
                    backbone.trainable = False
                except Exception:
                    for layer in probe_model.layers[:-1]:
                        layer.trainable = False

                probe_checkpoint = CBAM_TL_CHECKPOINTS_DIR / f"{model_name_with_tag}_img{img_size}_probe.weights.h5"
                probe_callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(filepath=str(probe_checkpoint), save_best_only=True, save_weights_only=True, verbose=0),
                ]
                probe_model.compile(optimizer=tf.keras.optimizers.Adam(PROBE_LR), loss="categorical_crossentropy", metrics=["accuracy"])                
                probe_model.fit(train_ds, validation_data=val_ds, epochs=PROBE_EPOCHS, class_weight=get_class_weight(y_train, num_classes), callbacks=probe_callbacks, verbose=1)

                before_metrics = evaluate_predictions(probe_model, test_ds, y_test)
                print(
                    "Before TL (probe) -> "
                    f"F1-macro: {before_metrics['f1_macro']:.4f} | "
                    f"F1-weighted: {before_metrics['f1_weighted']:.4f} | "
                    f"Acc: {before_metrics['accuracy']:.4f}"
                )

            # Prepare base model for TL (use a clone so original base remains untouched)
            base_for_tl = tf.keras.models.clone_model(base_model)
            base_for_tl.set_weights(base_model.get_weights())
            if not is_same_num_classes:
                base_for_tl = _rebuild_model_for_target_classes(base_for_tl, num_classes)

            class_weight = get_class_weight(y_train, num_classes)

            for strategy in STRATEGIES:
                print(f"\n--- Strategy: {strategy} ---")
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(BASE_SEED)

                model_ft = tf.keras.models.clone_model(base_for_tl)
                model_ft.set_weights(base_for_tl.get_weights())
                model_ft = apply_fine_tuning_strategy(model_ft, strategy)

                checkpoint_path = (
                    CBAM_TL_CHECKPOINTS_DIR
                    / f"{model_name_with_tag}_img{img_size}_{strategy}_best.weights.h5"
                )
                final_model_path = (
                    CBAM_TL_FINAL_MODEL_DIR / f"{model_name_with_tag}_img{img_size}_{strategy}.keras"
                )
                if final_model_path.exists():
                    print(f"Reusing trained TL model: {final_model_path.name}")
                    model_ft = tf.keras.models.load_model(
                        final_model_path, compile=False, custom_objects=_custom_objects()
                    )
                else:
                    train_tl(model_ft, train_ds, val_ds, class_weight, checkpoint_path)
                    model_ft.save(final_model_path)

                after_metrics = evaluate_predictions(model_ft, test_ds, y_test)
                print(
                    "After TL (target test) -> "
                    f"F1-macro: {after_metrics['f1_macro']:.4f} | "
                    f"F1-weighted: {after_metrics['f1_weighted']:.4f} | "
                    f"Acc: {after_metrics['accuracy']:.4f}"
                )
                y_prob = model_ft.predict(test_ds, verbose=0)
                y_pred = np.argmax(y_prob, axis=1)
                backbone_name = f"{model_name_with_tag}_img{img_size}_{strategy}"
                plot_and_save_confusion_matrix(y_test, y_pred, class_names, backbone_name, CBAM_TL_CONFUSION_MATRIX_DIR)
                print(f"Saved confusion matrix: {CBAM_TL_CONFUSION_MATRIX_DIR}/confusion_matrix_{backbone_name}.png")

                row = {
                    "model": model_name_with_tag,
                    "img_size": img_size,
                    "strategy": strategy,
                    "class_compatible": bool(is_same_num_classes),
                    "base_model_path": str(base_model_path),
                    "tl_model_path": str(final_model_path),
                }
                for metric_name, before_val in before_metrics.items():
                    row[f"before_{metric_name}"] = before_val
                    row[f"after_{metric_name}"] = after_metrics[metric_name]
                    row[f"delta_{metric_name}"] = after_metrics[metric_name] - before_val
                results = _upsert_result(results, row)

                pd.DataFrame(results).sort_values(
                    by="after_f1_macro", ascending=False
                ).to_csv(RESULTS_PATH, index=False)

    if not results:
        print(
            "No CBAM TL runs executed. Check availability of CBAM models in artifacts_cbam/final_model."
        )
        return

    df_results = pd.DataFrame(results).sort_values("after_f1_macro", ascending=False)
    df_best = (
        df_results.sort_values("after_f1_macro", ascending=False)
        .groupby(["model", "img_size"], as_index=False)
        .head(1)
        .sort_values("after_f1_macro", ascending=False)
    )

    df_results.to_csv(RESULTS_PATH, index=False)
    df_best.to_csv(BEST_RESULTS_PATH, index=False)

    print("\n=== CBAM TL ranking (all runs) ===")
    print(df_results.to_string(index=False))
    print("\n=== Best strategy per model|img_size ===")
    print(df_best.to_string(index=False))


if __name__ == "__main__":
    main()
