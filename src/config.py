"""Central place for paths and default hyperparameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PKL = ROOT / "data" / "raw" / "LSWMD.pkl"
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"

FAILURE_CLASSES = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]

# preprocessing
IMG_SIZE = 64
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SEED = 42

# training
BATCH_SIZE = 128
EPOCHS = 40  # upper bound; early stopping usually halts sooner
LR = 1e-3
EARLY_STOP_PATIENCE = 7  # stop after this many epochs with no val macro-F1 gain
LR_PATIENCE = 3  # epochs of no val macro-F1 gain before halving the LR
