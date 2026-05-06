import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import (
    apply_fine_tuning_strategy,
    build_training_progress,
    evaluate_predictions,
    find_backbone,
    get_class_weight,
    load_paths_and_labels,
    make_dataset,
    plot_and_save_confusion_matrix,
    rebuild_model_for_target_classes,
    set_global_seeds,
    train_tl,
    upsert_result,
    write_json,
)

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
PROBE_EPOCHS = 5
PROBE_LR = 1e-3
LAST_N_LAYERS = 20
FREEZE_BATCHNORM = True
USE_AUGMENTATION = True
USE_MIXED_PRECISION = True

BASE_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
BASE_FINAL_MODEL_DIR = BASE_ARTIFACTS_DIR / "final_model"

TL_ARTIFACTS_DIR = Path(os.environ.get("TL_ARTIFACTS_DIR", "artifacts_tl"))
TL_CHECKPOINTS_DIR = TL_ARTIFACTS_DIR / "checkpoints"
TL_FINAL_MODEL_DIR = TL_ARTIFACTS_DIR / "final_model"
TL_RESULTS_DIR = TL_ARTIFACTS_DIR / "results"
RESULTS_PATH = TL_RESULTS_DIR / "domain_shift_tl_results.csv"
BEST_RESULTS_PATH = TL_RESULTS_DIR / "domain_shift_tl_best_by_model.csv"
PROGRESS_STATE_PATH = TL_RESULTS_DIR / "tl_training_progress_state.json"
TL_CONFUSION_MATRIX_DIR = TL_ARTIFACTS_DIR / "confusionmatrix"

# =========================
# Model registry
# =========================
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


def _load_base_model(model_name, img_size):
    exact_path = BASE_FINAL_MODEL_DIR / f"{model_name}_img{img_size}_final.keras"
    if exact_path.exists():
        return tf.keras.models.load_model(exact_path, compile=False), exact_path

    best_path = BASE_FINAL_MODEL_DIR / f"{model_name}_best_final.keras"
    if best_path.exists():
        return tf.keras.models.load_model(best_path, compile=False), best_path

    return None, exact_path


def _compute_tl_progress_state(results):
    all_items = [
        f"{m}|img{s}|{st}"
        for m in MODELS
        for s in IMAGE_SIZES
        for st in STRATEGIES
    ]
    done_set = {
        f"{r['model']}|img{r['img_size']}|{r['strategy']}"
        for r in results
    }
    done_items = sorted(done_set)
    pending_items = [i for i in all_items if i not in done_set]
    return all_items, done_items, pending_items


def _persist_tl_progress_state(results, current_item=None):
    all_items, done_items, pending_items = _compute_tl_progress_state(results)
    state = build_training_progress(
        total_steps=len(all_items),
        done_items=done_items,
        pending_items=pending_items,
        current_item=current_item,
    )
    write_json(PROGRESS_STATE_PATH, state)
    print(f"TL Progress: {state['progress']} | pending: {state['pending_steps']}")


def main():
    set_global_seeds(BASE_SEED)
    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    X_target, y_target, class_names = load_paths_and_labels(TARGET_DATA_DIR)
    num_classes = len(class_names)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_target, y_target, test_size=TEST_SIZE, stratify=y_target, random_state=BASE_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, stratify=y_trainval, random_state=BASE_SEED,
    )

    print(f"Target domain classes: {class_names}")
    print(f"Target samples total: {len(X_target)}")
    print(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    TL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    TL_FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TL_CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH).to_dict(orient="records")
    _persist_tl_progress_state(results, current_item=None)

    for model_name in MODELS:
        _, preprocess_fn = MODEL_SPECS[model_name]
        for img_size in IMAGE_SIZES:
            print(f"\n===== {model_name} | img_size={img_size} =====")

            tf.keras.backend.clear_session()
            set_global_seeds(BASE_SEED)

            base_model, base_model_path = _load_base_model(model_name, img_size)
            if base_model is None:
                print(f"Skipping (base model not found): {base_model_path}")
                continue

            source_head_units = int(base_model.output_shape[-1])
            is_same_num_classes = source_head_units == num_classes

            train_ds = make_dataset(
                X_train, y_train, img_size, preprocess_fn, BATCH_SIZE,
                training=True, use_augmentation=USE_AUGMENTATION, seed=BASE_SEED,
            )
            val_ds = make_dataset(
                X_val, y_val, img_size, preprocess_fn, BATCH_SIZE,
                training=False, seed=BASE_SEED,
            )
            test_ds = make_dataset(
                X_test, y_test, img_size, preprocess_fn, BATCH_SIZE,
                training=False, seed=BASE_SEED,
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
                    f"Class mismatch for {model_name}|img{img_size}: "
                    f"source={source_head_units}, target={num_classes}. "
                    "Creating linear probe to estimate 'before' performance."
                )
                probe_base = tf.keras.models.clone_model(base_model)
                probe_base.set_weights(base_model.get_weights())
                probe_model = rebuild_model_for_target_classes(probe_base, num_classes)
                try:
                    find_backbone(probe_model).trainable = False
                except Exception:
                    for layer in probe_model.layers[:-1]:
                        layer.trainable = False

                probe_checkpoint = TL_CHECKPOINTS_DIR / f"{model_name}_img{img_size}_probe.weights.h5"
                probe_model.compile(
                    optimizer=tf.keras.optimizers.Adam(PROBE_LR),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"],
                )
                probe_model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=PROBE_EPOCHS,
                    class_weight=get_class_weight(y_train, num_classes),
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                        tf.keras.callbacks.ModelCheckpoint(filepath=str(probe_checkpoint), save_best_only=True, save_weights_only=True, verbose=0),
                    ],
                    verbose=1,
                )
                before_metrics = evaluate_predictions(probe_model, test_ds, y_test)
                print(
                    "Before TL (probe) -> "
                    f"F1-macro: {before_metrics['f1_macro']:.4f} | "
                    f"F1-weighted: {before_metrics['f1_weighted']:.4f} | "
                    f"Acc: {before_metrics['accuracy']:.4f}"
                )
                del probe_model, probe_base
                gc.collect()

            class_weight = get_class_weight(y_train, num_classes)
            base_for_tl = tf.keras.models.clone_model(base_model)
            base_for_tl.set_weights(base_model.get_weights())
            if not is_same_num_classes:
                base_for_tl = rebuild_model_for_target_classes(base_for_tl, num_classes)

            for strategy in STRATEGIES:
                print(f"\n--- Strategy: {strategy} ---")
                set_global_seeds(BASE_SEED)

                model_ft = tf.keras.models.clone_model(base_for_tl)
                model_ft.set_weights(base_for_tl.get_weights())
                model_ft = apply_fine_tuning_strategy(
                    model_ft, strategy,
                    last_n_layers=LAST_N_LAYERS,
                    freeze_batchnorm=FREEZE_BATCHNORM,
                )

                checkpoint_path = (
                    TL_CHECKPOINTS_DIR / f"{model_name}_img{img_size}_{strategy}_best.weights.h5"
                )
                final_model_path = (
                    TL_FINAL_MODEL_DIR / f"{model_name}_img{img_size}_{strategy}.keras"
                )
                if final_model_path.exists():
                    print(f"Reusing trained TL model: {final_model_path.name}")
                    del model_ft
                    gc.collect()
                    model_ft = tf.keras.models.load_model(final_model_path, compile=False)
                else:
                    train_tl(
                        model_ft, train_ds, val_ds, class_weight, checkpoint_path,
                        epochs=TL_EPOCHS, lr=TL_LR,
                    )
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
                del y_prob
                backbone_name = f"{model_name}_img{img_size}_{strategy}"
                plot_and_save_confusion_matrix(y_test, y_pred, class_names, backbone_name, TL_CONFUSION_MATRIX_DIR)
                print(f"Saved confusion matrix: {TL_CONFUSION_MATRIX_DIR}/confusion_matrix_{backbone_name}.png")

                row = {
                    "model": model_name,
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
                results = upsert_result(results, row)

                pd.DataFrame(results).sort_values(
                    by="after_f1_macro", ascending=False
                ).to_csv(RESULTS_PATH, index=False)
                _persist_tl_progress_state(results, current_item=f"{model_name}|img{img_size}|{strategy}")
                del model_ft
                gc.collect()

            del train_ds, val_ds, test_ds, base_for_tl, base_model
            gc.collect()
            tf.keras.backend.clear_session()

    if not results:
        print("No TL runs executed. Check availability of base models in artifacts/final_model.")
        return

    df_results = pd.DataFrame(results).sort_values("after_f1_macro", ascending=False)
    df_best = (
        df_results.groupby(["model", "img_size"], as_index=False)
        .head(1)
        .sort_values("after_f1_macro", ascending=False)
    )

    df_results.to_csv(RESULTS_PATH, index=False)
    df_best.to_csv(BEST_RESULTS_PATH, index=False)

    print("\n=== TL ranking (all runs) ===")
    print(df_results.to_string(index=False))
    print("\n=== Best strategy per model|img_size ===")
    print(df_best.to_string(index=False))
    _persist_tl_progress_state(results)


if __name__ == "__main__":
    main()
