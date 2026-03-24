from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils import load_paths_and_labels, make_dataset

# =========================
# Config
# =========================
DATA_DIR = Path("../data/public_dataset/public_data")
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

HEAD_EPOCHS = 100
FINETUNE_EPOCHS = 120
HEAD_LR = 1e-3
FINETUNE_LR = 1e-5
UNFREEZE_RATIO = 0.3

N_SPLITS = 5
N_REPEATS = 3
BATCH_SIZE = 32
BASE_SEED = 5681
TEST_SIZE = 0.2
IMAGE_SIZES = [192, 224, 320]

ARTIFACTS_DIR = Path("artifacts")
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FINAL_MODEL_DIR = ARTIFACTS_DIR / "final_model"

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


def _load_partial_metrics():
    fold_results = []
    results = []

    folds_csv = RESULTS_DIR / "cv_fold_metrics.csv"
    ranking_csv = RESULTS_DIR / "cv_ranking.csv"

    if folds_csv.exists():
        fold_results = pd.read_csv(folds_csv).to_dict(orient="records")
    if ranking_csv.exists():
        results = pd.read_csv(ranking_csv).to_dict(orient="records")
    return fold_results, results


def _persist_partial_metrics(fold_results, results):
    df_folds = pd.DataFrame(fold_results)
    df_results = pd.DataFrame(results)

    if not df_folds.empty:
        df_folds.to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    if not df_results.empty:
        df_results = df_results.sort_values("f1_macro_mean", ascending=False)
        df_results.to_csv(RESULTS_DIR / "cv_ranking.csv", index=False)


def _find_fold_result(fold_results, model_name, img_size, fold):
    return next(
        (
            row
            for row in fold_results
            if (
                row["model"] == model_name
                and int(row["img_size"]) == img_size
                and int(row["fold"]) == fold
            )
        ),
        None,
    )


def _upsert_fold_result(fold_results, row):
    filtered = [
        item
        for item in fold_results
        if not (
            item["model"] == row["model"]
            and int(item["img_size"]) == int(row["img_size"])
            and int(item["fold"]) == int(row["fold"])
        )
    ]
    filtered.append(row)
    return filtered


def build_model_with_base(model_name, num_classes, img_size, model_specs):
    base_cls, _ = model_specs[model_name]
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
    return model, base


def set_finetune_layers(base_model, unfreeze_ratio):
    base_model.trainable = True
    n_layers = len(base_model.layers)
    start_idx = int(n_layers * (1 - unfreeze_ratio))

    for layer in base_model.layers[:start_idx]:
        layer.trainable = False
    for layer in base_model.layers[start_idx:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def get_class_weight(labels, num_classes):
    cls_w = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    return {i: float(w) for i, w in enumerate(cls_w)}


def train_two_stage(
    model,
    base_model,
    train_ds,
    class_weight,
    val_ds=None,
    best_checkpoint_path=None,
):
    head_callbacks = []
    finetune_callbacks = []

    if val_ds is not None:
        head_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        )
        finetune_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            )
        )
        finetune_callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
            )
        )
        if best_checkpoint_path is not None:
            finetune_callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(best_checkpoint_path),
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=0,
                )
            )
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(HEAD_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weight,
        callbacks=head_callbacks,
        verbose=1,
    )

    set_finetune_layers(base_model, UNFREEZE_RATIO)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weight,
        callbacks=finetune_callbacks,
        verbose=1,
    )


def evaluate_predictions(model, dataset, y_true):
    y_prob = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    X, y, class_names = load_paths_and_labels(DATA_DIR)
    num_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=BASE_SEED,
    )

    print(f"Total samples: {len(X)}")
    print(f"Train samples:   {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fold_results, results = _load_partial_metrics()

    for model_name in MODELS:
        _, preprocess_fn = MODEL_SPECS[model_name]
        for img_size in IMAGE_SIZES:
            print(f"\n===== {model_name} | img_size={img_size} =====")

            rkf = RepeatedStratifiedKFold(
                n_splits=N_SPLITS,
                n_repeats=N_REPEATS,
                random_state=BASE_SEED,
            )

            fold_scores = []
            for fold, (tr_idx, va_idx) in enumerate(rkf.split(X_train, y_train), start=1):
                existing = _find_fold_result(fold_results, model_name, img_size, fold)
                checkpoint_path = (
                    CHECKPOINTS_DIR
                    / f"{model_name}_img{img_size}_fold{fold:02d}_best.weights.h5"
                )
                if existing is not None:
                    print(
                        f"Skipping completed fold with saved metrics: {model_name}|img{img_size}|fold{fold:02d}"
                    )
                    fold_scores.append(
                        {
                            "f1_macro": float(existing["f1_macro"]),
                            "f1_weighted": float(existing["f1_weighted"]),
                            "accuracy": float(existing["accuracy"]),
                            "precision_macro": float(existing["precision_macro"]),
                            "recall_macro": float(existing["recall_macro"]),
                        }
                    )
                    continue

                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(BASE_SEED + fold)

                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]

                train_ds = make_dataset(
                    X_tr, y_tr, img_size, preprocess_fn, BATCH_SIZE, training=True
                )
                val_ds = make_dataset(
                    X_va, y_va, img_size, preprocess_fn, BATCH_SIZE, training=False
                )

                class_weight = get_class_weight(y_tr, num_classes)
                model, base = build_model_with_base(
                    model_name, num_classes, img_size, MODEL_SPECS
                )
                if checkpoint_path.exists():
                    print(
                        f"Skipping training for fold with existing checkpoint: {checkpoint_path.name}"
                    )
                    model.load_weights(checkpoint_path)
                else:
                    train_two_stage(
                        model=model,
                        base_model=base,
                        train_ds=train_ds,
                        class_weight=class_weight,
                        val_ds=val_ds,
                        best_checkpoint_path=checkpoint_path,
                    )

                fold_metrics = evaluate_predictions(model, val_ds, y_va)
                fold_scores.append(fold_metrics)
                fold_results = _upsert_fold_result(
                    fold_results,
                    {
                        "model": model_name,
                        "img_size": img_size,
                        "fold": fold,
                        **fold_metrics,
                    },
                )
                print(
                    "Fold "
                    f"{fold:02d} -> "
                    f"F1-macro: {fold_metrics['f1_macro']:.4f} | "
                    f"F1-weighted: {fold_metrics['f1_weighted']:.4f} | "
                    f"Acc: {fold_metrics['accuracy']:.4f} | "
                    f"Prec-macro: {fold_metrics['precision_macro']:.4f} | "
                    f"Recall-macro: {fold_metrics['recall_macro']:.4f}"
                )
                _persist_partial_metrics(fold_results, results)

            f1_macros = [s["f1_macro"] for s in fold_scores]
            f1_weighted = [s["f1_weighted"] for s in fold_scores]
            accs = [s["accuracy"] for s in fold_scores]
            precision_macros = [s["precision_macro"] for s in fold_scores]
            recall_macros = [s["recall_macro"] for s in fold_scores]

            print(
                f"{model_name} (img={img_size}) | F1-macro mean±std: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}"
            )
            print(
                f"{model_name} (img={img_size}) | F1-weighted mean±std: {np.mean(f1_weighted):.4f} ± {np.std(f1_weighted):.4f}"
            )
            print(
                f"{model_name} (img={img_size}) | Acc mean±std:      {np.mean(accs):.4f} ± {np.std(accs):.4f}"
            )
            print(
                f"{model_name} (img={img_size}) | Prec-macro mean±std: {np.mean(precision_macros):.4f} ± {np.std(precision_macros):.4f}"
            )
            print(
                f"{model_name} (img={img_size}) | Recall-macro mean±std: {np.mean(recall_macros):.4f} ± {np.std(recall_macros):.4f}"
            )

            summary = {
                "model": model_name,
                "img_size": img_size,
                "f1_macro_mean": np.mean(f1_macros),
                "f1_macro_std": np.std(f1_macros),
                "f1_weighted_mean": np.mean(f1_weighted),
                "f1_weighted_std": np.std(f1_weighted),
                "acc_mean": np.mean(accs),
                "acc_std": np.std(accs),
                "precision_macro_mean": np.mean(precision_macros),
                "precision_macro_std": np.std(precision_macros),
                "recall_macro_mean": np.mean(recall_macros),
                "recall_macro_std": np.std(recall_macros),
                "runs": len(fold_scores),
            }
            results = [
                row
                for row in results
                if not (row["model"] == model_name and int(row["img_size"]) == img_size)
            ]
            results.append(summary)
            _persist_partial_metrics(fold_results, results)

    df_folds = pd.DataFrame(fold_results)
    df_results = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
    print("\n=== Final ranking ===")
    print(df_results.to_string(index=False))
    df_folds.to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    df_results.to_csv(RESULTS_DIR / "cv_ranking.csv", index=False)

    best_model_name = df_results.iloc[0]["model"]
    best_img_size = int(df_results.iloc[0]["img_size"])
    _, best_preprocess_fn = MODEL_SPECS[best_model_name]
    print(
        f"\n=== Final model selected from CV: {best_model_name} | img_size={best_img_size} ==="
    )

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(BASE_SEED)

    train_ds = make_dataset(
        X_train, y_train, best_img_size, best_preprocess_fn, BATCH_SIZE, training=True
    )
    test_ds = make_dataset(
        X_test, y_test, best_img_size, best_preprocess_fn, BATCH_SIZE, training=False
    )

    class_weight = get_class_weight(y_train, num_classes)
    final_model, final_base = build_model_with_base(
        best_model_name, num_classes, best_img_size, MODEL_SPECS
    )

    final_model_path = FINAL_MODEL_DIR / f"{best_model_name}_img{best_img_size}_final.keras"
    holdout_metrics_path = RESULTS_DIR / "holdout_test_metrics.csv"
    if final_model_path.exists() and holdout_metrics_path.exists():
        print("Final stage already completed. Skipping retraining and holdout evaluation.")
    else:
        train_two_stage(
            model=final_model,
            base_model=final_base,
            train_ds=train_ds,
            class_weight=class_weight,
            val_ds=None,
            best_checkpoint_path=None,
        )

        final_model.save(final_model_path)

        test_metrics = evaluate_predictions(final_model, test_ds, y_test)
        print(
            "Holdout TEST -> "
            f"F1-macro: {test_metrics['f1_macro']:.4f} | "
            f"F1-weighted: {test_metrics['f1_weighted']:.4f} | "
            f"Acc: {test_metrics['accuracy']:.4f} | "
            f"Prec-macro: {test_metrics['precision_macro']:.4f} | "
            f"Recall-macro: {test_metrics['recall_macro']:.4f}"
        )
        pd.DataFrame(
            [
                {
                    "model": best_model_name,
                    "img_size": best_img_size,
                    **test_metrics,
                }
            ]
        ).to_csv(holdout_metrics_path, index=False)


if __name__ == "__main__":
    main()
