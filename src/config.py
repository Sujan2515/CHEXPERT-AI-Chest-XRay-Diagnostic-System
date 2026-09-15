from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directories
DATASET_ROOT = PROJECT_ROOT / "Datasets" / "raw"

PNEUMONIA_ROOT = DATASET_ROOT / "Pneumonia" / "chest_xray" / "chest_xray"

TB_ROOT = DATASET_ROOT / "Tuberculosis" / "TB_Chest_Radiography_Database"

# Model storage
MODEL_DIR = PROJECT_ROOT / "models"

# Create model directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 224

# Number of images processed at once
BATCH_SIZE = 8


# ============================================================
# TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 10

LEARNING_RATE = 0.0001

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL FILES
# ============================================================

PNEUMONIA_MODEL_PATH = MODEL_DIR / "pneumonia_model.pth"

TB_MODEL_PATH = MODEL_DIR / "tb_model.pth"


# ============================================================
# PRINT CONFIGURATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHEXPERT PROJECT CONFIGURATION")
    print("=" * 60)

    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Dataset root   : {DATASET_ROOT}")
    print(f"Pneumonia root : {PNEUMONIA_ROOT}")
    print(f"TB root        : {TB_ROOT}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Image size     : {IMAGE_SIZE}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Epochs         : {NUM_EPOCHS}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print(f"Device         : {DEVICE}")