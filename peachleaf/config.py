"""Single source of truth for paths, class definitions and hyper-parameters.

Every experiment reads its settings from here so that the manuscript (Table 4
and Sections 2.4-2.7) can be regenerated from one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PEACH_DATA_ROOT", REPO_ROOT / "data"))
PUBLIC_DIR = DATA_ROOT / "public_dataset" / "public_data"
LOCAL_DIR = DATA_ROOT / "merged_dataset" / "local_data"

SPLITS_DIR = REPO_ROOT / "splits"
MANIFEST_CSV = SPLITS_DIR / "manifest.csv"
RUNS_ROOT = Path(os.environ.get("PEACH_RUNS_ROOT", REPO_ROOT / "runs"))
REPORTS_ROOT = Path(os.environ.get("PEACH_REPORTS_ROOT", REPO_ROOT / "reports"))

# -----------------------------------------------------------------------------
# Label space
# -----------------------------------------------------------------------------
# A single six-way label space is shared by both domains.  Local leaves use the
# same indices, so a source model never needs its head replaced.
CLASS_NAMES = [
    "abiotic_stress",
    "bacterial_spot",
    "chewing_insect",
    "healthy",
    "mechanical_stress",
    "mite_presence",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
LOCAL_CLASSES = ["abiotic_stress", "chewing_insect", "healthy", "mechanical_stress"]
LOCAL_CLASS_IDX = [CLASS_TO_IDX[c] for c in LOCAL_CLASSES]
HEALTHY_IDX = CLASS_TO_IDX["healthy"]

# -----------------------------------------------------------------------------
# Architectures
# -----------------------------------------------------------------------------
BACKBONES = [
    "mobilenetv2",
    "mobilenetv3large",
    "efficientnetb0",
    "densenet121",
    "efficientnetb3",
    "resnet50",
    "inceptionv3",
    "vgg16",
    "vgg19",
    "efficientnetb5",
    "resnet101",
]
# "none" is the plain backbone; "cbam" is the terminal-stage CBAM used in the
# main comparison.  The remaining variants are ablations.
ATTENTION_VARIANTS = ["none", "cbam", "se", "cbam_channel", "cbam_spatial", "cbam_2stage"]
MAIN_VARIANTS = ["none", "cbam"]
ABLATION_VARIANTS = ["se", "cbam_channel", "cbam_spatial", "cbam_2stage"]
# One light, one medium and one large backbone, chosen a priori by size class
# (not by performance), so the choice does not depend on the results.
ABLATION_BACKBONES = ["mobilenetv2", "efficientnetb3", "resnet50"]
CBAM_REDUCTION = 16

IMG_SIZE = 224  # fixed for every model

# -----------------------------------------------------------------------------
# Splitting protocol
# -----------------------------------------------------------------------------
SPLIT_SEED = 2026
PUBLIC_TEST_FRACTION_FOLDS = 5  # one of five grouped folds -> ~20 % hold-out
CV_SPLITS = 5
CV_REPEATS = 3
LOCAL_SPLITS = 5
LOCAL_REPEATS = 3
INNER_VAL_FRACTION = 0.2  # grouped validation split inside each local train fold

# -----------------------------------------------------------------------------
# Source-domain training (Table 4)
# -----------------------------------------------------------------------------
BATCH_SIZE = 16
HEAD_EPOCHS = 100
HEAD_LR = 1e-3
HEAD_PATIENCE = 6
FINETUNE_EPOCHS = 120
FINETUNE_LR = 1e-5
FINETUNE_PATIENCE = 10
LR_PATIENCE = 3  # strictly smaller than the early-stopping patience
LR_FACTOR = 0.5
MIN_LR = 1e-7
UNFREEZE_FRACTION = 0.3  # top 30 % of backbone *layers* (BatchNorm kept frozen)
DROPOUT = 0.3
MONITOR = "val_loss"

# -----------------------------------------------------------------------------
# Target-domain adaptation
# -----------------------------------------------------------------------------
TL_STRATEGIES = ["lp", "ftl", "fta", "lpft"]
LP_LR = 1e-3
LP_EPOCHS = 60
FT_LR = 5e-5
FT_EPOCHS = 60
TL_PATIENCE = 8
TL_LR_PATIENCE = 3
WISE_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
LEARNING_CURVE_SIZES = [10, 25, 50, 100]  # plus "all"
JOINT_BACKBONES = ["mobilenetv3large", "efficientnetb0", "densenet121", "efficientnetb3"]
CURVE_BACKBONES = JOINT_BACKBONES

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
MIXED_PRECISION = os.environ.get("PEACH_MIXED_PRECISION", "1") == "1"
SAVE_MODEL_REPEATS = {1}  # source CV models kept on disk (used by the TL stage)


def seed_for(repeat: int, fold: int, offset: int = 0) -> int:
    """Deterministic, unique seed for a (repeat, fold) unit."""
    return 1000 * int(repeat) + int(fold) + offset

# Smoke-test mode: one epoch everywhere (used to validate the pipeline end to end)
if os.environ.get("PEACH_SMOKE") == "1":
    HEAD_EPOCHS = FINETUNE_EPOCHS = LP_EPOCHS = FT_EPOCHS = 1
