from pathlib import Path
import random
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TB_ROOT = (
    PROJECT_ROOT
    / "Datasets"
    / "raw"
    / "Tuberculosis"
    / "TB_Chest_Radiography_Database"
    / "Tuberculosis"
)

COMMON_ROOT = PROJECT_ROOT / "Datasets" / "common"

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

TRAIN_COUNT = 490
VAL_COUNT = 105
TEST_COUNT = 105

TOTAL_COUNT = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

random.seed(SEED)


# ============================================================
# GET TB IMAGES
# ============================================================

print("=" * 60)
print("ADDING TB IMAGES TO COMMON DATASET")
print("=" * 60)

print("\nShenzhen dataset: NOT USED")
print("Original TB dataset: USED")
print("Original images: COPIED, not moved")


if not TB_ROOT.exists():
    raise FileNotFoundError(
        f"TB dataset folder not found:\n{TB_ROOT}"
    )


tb_images = [
    path
    for path in TB_ROOT.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"\nTB images found: {len(tb_images)}")


if len(tb_images) != TOTAL_COUNT:
    raise ValueError(
        f"Expected exactly {TOTAL_COUNT} TB images, "
        f"but found {len(tb_images)}."
    )


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(tb_images)

train_images = tb_images[:TRAIN_COUNT]

val_images = tb_images[
    TRAIN_COUNT:
    TRAIN_COUNT + VAL_COUNT
]

test_images = tb_images[
    TRAIN_COUNT + VAL_COUNT:
]


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(images, destination, prefix):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, source in enumerate(images, start=1):

        destination_path = (
            destination
            / f"{prefix}_{index:04d}{source.suffix.lower()}"
        )

        shutil.copy2(
            source,
            destination_path
        )


# ============================================================
# COPY TB IMAGES
# ============================================================

print("\nCopying TB training images...")
copy_images(
    train_images,
    COMMON_ROOT / "train" / "Tuberculosis",
    "tb_train"
)

print("Copying TB validation images...")
copy_images(
    val_images,
    COMMON_ROOT / "val" / "Tuberculosis",
    "tb_val"
)

print("Copying TB test images...")
copy_images(
    test_images,
    COMMON_ROOT / "test" / "Tuberculosis",
    "tb_test"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TB COMMON DATASET COMPLETED")
print("=" * 60)

print(f"Training      : {len(train_images)}")
print(f"Validation    : {len(val_images)}")
print(f"Testing       : {len(test_images)}")
print(f"Total TB      : {len(tb_images)}")

print("\nCOMMON DATASET TOTAL")
print("--------------------")
print("Normal        : 1000")
print("Pneumonia     : 1000")
print("Tuberculosis  : 700")
print("--------------------")
print("TOTAL         : 2700")

print("\nSplit totals:")
print("Train         : 1890")
print("Validation    : 405")
print("Test          : 405")

print("\n✓ TB images successfully added.")
print("✓ Shenzhen dataset was NOT used.")
print("✓ Current training pipeline remains unchanged.")