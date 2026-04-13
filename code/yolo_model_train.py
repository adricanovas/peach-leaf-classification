from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from utils import (
    build_training_progress,
    load_paths_and_labels,
    write_json,
)

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics is required for yolo_model_train.py. Install it with: pip install ultralytics"
    ) from exc

# =========================
# Config
# =========================
DATA_DIR = Path("../data/public_dataset/public_data")
YOLO_MODELS = {
    "yolo11n-cls.pt": {"batch": 32},
    "yolo11m-cls.pt": {"batch": 32},
    "yolo11l-cls.pt": {"batch": 16},
}
IMAGE_SIZES = [192, 224]

EPOCHS = 30
PATIENCE = 8
N_SPLITS = 5
N_REPEATS = 3
BASE_SEED = 5681
TEST_SIZE = 0.2
FINAL_VAL_SIZE = 0.1
WORKERS = 4
DEVICE = 0

ARTIFACTS_DIR = Path("artifacts_yolo")
#ARTIFACTS_DIR = Path("artifacts_yolo")
RUNS_DIR = ARTIFACTS_DIR / "runs"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FINAL_MODEL_DIR = ARTIFACTS_DIR / "final_model"

PROGRESS_STATE_PATH = RESULTS_DIR / "training_progress_state.json"
HOLDOUT_ALL_MODELS_PATH = RESULTS_DIR / "holdout_test_metrics_all_models.csv"
CV_FOLD_METRICS_PATH = RESULTS_DIR / "cv_fold_metrics.csv"
CV_RANKING_PATH = RESULTS_DIR / "cv_ranking.csv"
HOLDOUT_BEST_PATH = RESULTS_DIR / "holdout_test_metrics.csv"
BEST_PER_ARCH_PATH = RESULTS_DIR / "best_model_per_architecture.csv"


def _sanitize_name(name: str) -> str:
    return name.replace(".pt", "").replace("-", "_")


def _fold_item_key(model_name: str, img_size: int, fold: int) -> str:
    return f"{model_name}|img{img_size}|fold{int(fold):02d}"


def _load_partial_metrics() -> Tuple[List[Dict], List[Dict]]:
    fold_results: List[Dict] = []
    results: List[Dict] = []

    if CV_FOLD_METRICS_PATH.exists():
        fold_results = pd.read_csv(CV_FOLD_METRICS_PATH).to_dict(orient="records")
    if CV_RANKING_PATH.exists():
        results = pd.read_csv(CV_RANKING_PATH).to_dict(orient="records")
    return fold_results, results


def _persist_partial_metrics(fold_results: List[Dict], results: List[Dict]) -> None:
    df_folds = pd.DataFrame(fold_results)
    df_results = pd.DataFrame(results)

    if not df_folds.empty:
        df_folds.to_csv(CV_FOLD_METRICS_PATH, index=False)
    if not df_results.empty:
        df_results = df_results.sort_values("f1_macro_mean", ascending=False)
        df_results.to_csv(CV_RANKING_PATH, index=False)


def _load_holdout_metrics_all_models() -> List[Dict]:
    if HOLDOUT_ALL_MODELS_PATH.exists():
        return pd.read_csv(HOLDOUT_ALL_MODELS_PATH).to_dict(orient="records")
    return []


def _find_fold_result(
    fold_results: Sequence[Dict], model_name: str, img_size: int, fold: int
) -> Dict | None:
    return next(
        (
            row
            for row in fold_results
            if (
                row["model"] == model_name
                and int(row["img_size"]) == int(img_size)
                and int(row["fold"]) == int(fold)
            )
        ),
        None,
    )


def _upsert_fold_result(fold_results: List[Dict], row: Dict) -> List[Dict]:
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


def _upsert_model_metric(rows: List[Dict], row: Dict) -> List[Dict]:
    filtered = [
        item
        for item in rows
        if not (
            item["model"] == row["model"]
            and int(item["img_size"]) == int(row["img_size"])
        )
    ]
    filtered.append(row)
    return filtered


def _compute_fold_progress_state(fold_results: Sequence[Dict]) -> Tuple[List[str], List[str], List[str]]:
    all_items = [
        _fold_item_key(model_name, img_size, fold)
        for model_name in YOLO_MODELS
        for img_size in IMAGE_SIZES
        for fold in range(1, (N_SPLITS * N_REPEATS) + 1)
    ]
    done_items = sorted(
        {
            _fold_item_key(row["model"], int(row["img_size"]), int(row["fold"]))
            for row in fold_results
        }
    )
    done_set = set(done_items)
    pending_items = [item for item in all_items if item not in done_set]
    return all_items, done_items, pending_items


def _persist_progress_state(fold_results: Sequence[Dict], current_item: str | None = None) -> None:
    all_items, done_items, pending_items = _compute_fold_progress_state(fold_results)
    state = build_training_progress(
        total_steps=len(all_items),
        done_items=done_items,
        pending_items=pending_items,
        current_item=current_item,
    )
    write_json(PROGRESS_STATE_PATH, state)
    print(f"Progress: {state['progress']} | pending: {state['pending_steps']}")


def _sha1_short(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _read_json(path: Path) -> Dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Could not read JSON file {path}: {exc}. Rebuilding it.")
        return None


def _split_stats(labels: Sequence[int], class_names: Sequence[str]) -> Dict[str, int]:
    counts = {name: 0 for name in class_names}
    for label in labels:
        counts[class_names[int(label)]] += 1
    return counts


def _expected_manifest(
    split_payload: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_names: Sequence[str],
) -> Dict:
    payload = {}
    for split_name, (_, labels) in split_payload.items():
        payload[split_name] = {
            "total": int(len(labels)),
            "per_class": _split_stats(labels, class_names),
        }
    return payload


def _prepare_yolo_cls_dataset(
    dataset_root: Path,
    split_payload: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_names: Sequence[str],
) -> None:
    manifest_path = dataset_root / "manifest.json"
    expected = _expected_manifest(split_payload, class_names)
    current_manifest = _read_json(manifest_path)

    if current_manifest is not None and current_manifest.get("splits") == expected:
        missing_split_dir = any(not (dataset_root / split).exists() for split in split_payload)
        if not missing_split_dir:
            return

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    for split_name in split_payload:
        for class_name in class_names:
            (dataset_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    for split_name, (paths, labels) in split_payload.items():
        for src_path, label in zip(paths, labels):
            src = Path(str(src_path))
            label_idx = int(label)
            class_name = class_names[label_idx]
            suffix = src.suffix.lower() if src.suffix else ".jpg"
            dst_name = f"{src.stem}_{_sha1_short(str(src.resolve()))}{suffix}"
            dst = dataset_root / split_name / class_name / dst_name
            _link_or_copy(src, dst)

    write_json(
        manifest_path,
        {
            "splits": expected,
        },
    )


def _get_run_paths(run_name: str) -> Tuple[Path, Path, Path]:
    run_dir = RUNS_DIR / run_name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    return run_dir, best_path, last_path


def _train_or_resume(
    model_source: str,
    run_name: str,
    dataset_root: Path,
    img_size: int,
    batch_size: int,
    seed: int,
) -> Path:
    run_dir, best_path, last_path = _get_run_paths(run_name)
    if best_path.exists():
        print(f"Skipping training: best checkpoint already exists ({best_path})")
        return best_path

    train_params = {
        "data": str(dataset_root),
        "epochs": EPOCHS,
        "imgsz": int(img_size),
        "batch": int(batch_size),
        "project": str(RUNS_DIR),
        "name": run_name,
        "exist_ok": True,
        "workers": WORKERS,
        "patience": PATIENCE,
        "device": DEVICE,
        "seed": int(seed),
    }

    if last_path.exists():
        print(f"Found interrupted checkpoint, trying resume: {last_path}")
        try:
            YOLO(str(last_path)).train(resume=True)
            if best_path.exists():
                return best_path
        except Exception as exc:
            print(f"[WARN] Resume failed for {run_name}: {exc}. Falling back to warm-start.")

        warm_start_source = str(last_path)
        print(f"Warm-starting training from {warm_start_source}")
        YOLO(warm_start_source).train(**train_params)
    else:
        print(f"Training from pretrained source: {model_source}")
        YOLO(model_source).train(**train_params)

    if best_path.exists():
        return best_path
    if last_path.exists():
        print(f"[WARN] best.pt not found in {run_dir}, using last.pt for evaluation.")
        return last_path
    raise FileNotFoundError(f"No checkpoint produced for run: {run_name}")


def _extract_class_names(model: YOLO) -> List[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[idx]) for idx in sorted(names.keys())]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def _evaluate_predictions(
    model: YOLO,
    image_paths: Sequence[str],
    y_true: np.ndarray,
    img_size: int,
    batch_size: int,
    expected_class_names: Sequence[str],
) -> Dict[str, float]:
    model_class_names = _extract_class_names(model)
    if model_class_names and list(model_class_names) != list(expected_class_names):
        print(
            "[WARN] Model class mapping differs from dataset class_names. "
            "Metrics are still computed with numeric labels; verify class order."
        )
        print(f"       model names:   {model_class_names}")
        print(f"       dataset names: {list(expected_class_names)}")

    y_pred: List[int] = []
    n_samples = len(image_paths)

    for start in range(0, n_samples, batch_size):
        batch = [str(p) for p in image_paths[start : start + batch_size]]
        results = model.predict(source=batch, imgsz=img_size, verbose=False, stream=False)
        for result in results:
            if result.probs is None:
                raise RuntimeError("YOLO prediction returned no class probabilities.")
            y_pred.append(int(result.probs.top1))

    y_pred_np = np.array(y_pred, dtype=np.int32)
    return {
        "f1_macro": float(f1_score(y_true, y_pred_np, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred_np, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred_np)),
        "precision_macro": float(
            precision_score(y_true, y_pred_np, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred_np, average="macro", zero_division=0)
        ),
    }


def _ensure_final_model_and_holdout(
    model_source: str,
    img_size: int,
    batch_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    holdout_all: List[Dict],
) -> List[Dict]:
    model_key = model_source.replace(".pt", "")
    model_safe = _sanitize_name(model_key)
    final_model_path = FINAL_MODEL_DIR / f"{model_key}_img{img_size}_final.pt"
    existing_holdout = next(
        (
            item
            for item in holdout_all
            if item["model"] == model_key and int(item["img_size"]) == int(img_size)
        ),
        None,
    )

    print(f"\n=== Final full-train: {model_key} | img_size={img_size} ===")

    if final_model_path.exists() and existing_holdout is not None:
        print("Final model and holdout metrics already exist. Skipping.")
        return holdout_all

    try:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=FINAL_VAL_SIZE,
            stratify=y_train,
            random_state=BASE_SEED,
        )
    except ValueError:
        print(
            "[WARN] Stratified final validation split failed. "
            "Using random split without stratify."
        )
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=FINAL_VAL_SIZE,
            random_state=BASE_SEED,
        )

    final_dataset_root = DATASETS_DIR / "final" / f"{model_safe}_img{img_size}"
    _prepare_yolo_cls_dataset(
        dataset_root=final_dataset_root,
        split_payload={
            "train": (X_fit, y_fit),
            "val": (X_val, y_val),
        },
        class_names=class_names,
    )

    if final_model_path.exists():
        print(f"Reusing final model: {final_model_path}")
        eval_model = YOLO(str(final_model_path))
    else:
        run_name = f"{model_safe}_img{img_size}_final"
        best_ckpt = _train_or_resume(
            model_source=model_source,
            run_name=run_name,
            dataset_root=final_dataset_root,
            img_size=img_size,
            batch_size=batch_size,
            seed=BASE_SEED,
        )
        shutil.copy2(best_ckpt, final_model_path)
        print(f"Saved final model: {final_model_path}")
        eval_model = YOLO(str(final_model_path))

    test_metrics = _evaluate_predictions(
        model=eval_model,
        image_paths=[str(p) for p in X_test],
        y_true=y_test,
        img_size=img_size,
        batch_size=batch_size,
        expected_class_names=class_names,
    )
    print(
        "Holdout TEST -> "
        f"F1-macro: {test_metrics['f1_macro']:.4f} | "
        f"F1-weighted: {test_metrics['f1_weighted']:.4f} | "
        f"Acc: {test_metrics['accuracy']:.4f} | "
        f"Prec-macro: {test_metrics['precision_macro']:.4f} | "
        f"Recall-macro: {test_metrics['recall_macro']:.4f}"
    )

    holdout_all = _upsert_model_metric(
        holdout_all,
        {
            "model": model_key,
            "img_size": int(img_size),
            **test_metrics,
        },
    )
    pd.DataFrame(holdout_all).to_csv(HOLDOUT_ALL_MODELS_PATH, index=False)
    return holdout_all


def _build_fold_indices(y_train: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    rkf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=BASE_SEED,
    )
    fold_indices: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for fold, (tr_idx, va_idx) in enumerate(rkf.split(np.zeros(len(y_train)), y_train), start=1):
        fold_indices.append((fold, tr_idx, va_idx))
    return fold_indices


def main() -> None:
    X, y, class_names = load_paths_and_labels(DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=BASE_SEED,
    )

    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fold_results, results = _load_partial_metrics()
    holdout_all = _load_holdout_metrics_all_models()
    _persist_progress_state(fold_results, current_item=None)

    fold_indices = _build_fold_indices(y_train)
    prepared_fold_dirs: Dict[int, Path] = {}

    for model_source, model_cfg in YOLO_MODELS.items():
        model_key = model_source.replace(".pt", "")
        model_safe = _sanitize_name(model_key)
        batch_size = int(model_cfg["batch"])

        for img_size in IMAGE_SIZES:
            print(f"\n===== {model_key} | img_size={img_size} =====")
            fold_scores: List[Dict[str, float]] = []

            for fold, tr_idx, va_idx in fold_indices:
                fold_item = _fold_item_key(model_source, img_size, fold)
                existing = _find_fold_result(fold_results, model_source, img_size, fold)
                if existing is not None:
                    print(
                        f"Skipping completed fold with saved metrics: "
                        f"{model_key}|img{img_size}|fold{fold:02d}"
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
                    _persist_progress_state(fold_results, current_item=fold_item)
                    continue

                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]

                if fold not in prepared_fold_dirs:
                    fold_dataset_root = DATASETS_DIR / "cv" / f"fold_{fold:02d}"
                    _prepare_yolo_cls_dataset(
                        dataset_root=fold_dataset_root,
                        split_payload={
                            "train": (X_tr, y_tr),
                            "val": (X_va, y_va),
                        },
                        class_names=class_names,
                    )
                    prepared_fold_dirs[fold] = fold_dataset_root

                run_name = f"{model_safe}_img{img_size}_fold{fold:02d}"
                best_ckpt = _train_or_resume(
                    model_source=model_source,
                    run_name=run_name,
                    dataset_root=prepared_fold_dirs[fold],
                    img_size=img_size,
                    batch_size=batch_size,
                    seed=BASE_SEED + fold,
                )

                fold_model = YOLO(str(best_ckpt))
                fold_metrics = _evaluate_predictions(
                    model=fold_model,
                    image_paths=[str(p) for p in X_va],
                    y_true=y_va,
                    img_size=img_size,
                    batch_size=batch_size,
                    expected_class_names=class_names,
                )
                fold_scores.append(fold_metrics)

                fold_results = _upsert_fold_result(
                    fold_results,
                    {
                        "model": model_source,
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
                _persist_progress_state(fold_results, current_item=fold_item)

            f1_macros = [s["f1_macro"] for s in fold_scores]
            f1_weighted = [s["f1_weighted"] for s in fold_scores]
            accs = [s["accuracy"] for s in fold_scores]
            precision_macros = [s["precision_macro"] for s in fold_scores]
            recall_macros = [s["recall_macro"] for s in fold_scores]

            print(
                f"{model_key} (img={img_size}) | "
                f"F1-macro mean±std: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}"
            )
            print(
                f"{model_key} (img={img_size}) | "
                f"F1-weighted mean±std: {np.mean(f1_weighted):.4f} ± {np.std(f1_weighted):.4f}"
            )
            print(
                f"{model_key} (img={img_size}) | "
                f"Acc mean±std: {np.mean(accs):.4f} ± {np.std(accs):.4f}"
            )
            print(
                f"{model_key} (img={img_size}) | "
                f"Prec-macro mean±std: {np.mean(precision_macros):.4f} ± {np.std(precision_macros):.4f}"
            )
            print(
                f"{model_key} (img={img_size}) | "
                f"Recall-macro mean±std: {np.mean(recall_macros):.4f} ± {np.std(recall_macros):.4f}"
            )

            summary = {
                "model": model_source,
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
                if not (
                    row["model"] == summary["model"]
                    and int(row["img_size"]) == int(summary["img_size"])
                )
            ]
            results.append(summary)
            _persist_partial_metrics(fold_results, results)
            _persist_progress_state(fold_results, current_item=None)

            holdout_all = _ensure_final_model_and_holdout(
                model_source=model_source,
                img_size=img_size,
                batch_size=batch_size,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                class_names=class_names,
                holdout_all=holdout_all,
            )

    df_folds = pd.DataFrame(fold_results)
    df_results = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
    if df_results.empty:
        print("No YOLO runs completed successfully. Check training logs in artifacts_yolo/runs.")
        return

    print("\n=== Final ranking ===")
    print(df_results.to_string(index=False))

    df_folds.to_csv(CV_FOLD_METRICS_PATH, index=False)
    df_results.to_csv(CV_RANKING_PATH, index=False)

    for row in df_results.to_dict(orient="records"):
        holdout_all = _ensure_final_model_and_holdout(
            model_source=row["model"],
            img_size=int(row["img_size"]),
            batch_size=int(YOLO_MODELS[row["model"]]["batch"]),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            holdout_all=holdout_all,
        )

    if holdout_all:
        best_model_name = df_results.iloc[0]["model"]
        best_img_size = int(df_results.iloc[0]["img_size"])
        best_holdout = next(
            (
                item
                for item in holdout_all
                if item["model"] == best_model_name
                and int(item["img_size"]) == best_img_size
            ),
            None,
        )
        if best_holdout is not None:
            pd.DataFrame([best_holdout]).to_csv(HOLDOUT_BEST_PATH, index=False)

    best_per_arch_rows = []
    for row in (
        df_results.sort_values("f1_macro_mean", ascending=False)
        .groupby("model", as_index=False)
        .head(1)
        .to_dict(orient="records")
    ):
        model_key = row["model"].replace(".pt", "")
        img_size = int(row["img_size"])
        src_path = FINAL_MODEL_DIR / f"{model_key}_img{img_size}_final.pt"
        dst_path = FINAL_MODEL_DIR / f"{model_key}_best_final.pt"
        if not src_path.exists():
            print(
                f"[WARN] Best candidate missing on disk, skipping best alias: {src_path.name}"
            )
            continue
        shutil.copy2(src_path, dst_path)
        best_per_arch_rows.append(
            {
                "model": row["model"],
                "best_img_size": img_size,
                "f1_macro_mean": float(row["f1_macro_mean"]),
                "source_model_path": str(src_path),
                "best_model_path": str(dst_path),
            }
        )

    if best_per_arch_rows:
        pd.DataFrame(best_per_arch_rows).sort_values(
            "f1_macro_mean", ascending=False
        ).to_csv(BEST_PER_ARCH_PATH, index=False)


if __name__ == "__main__":
    main()
