from pathlib import Path
import random
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PNEUMONIA_ROOT = (
    PROJECT_ROOT
    / "Datasets"
    / "raw"
    / "Pneumonia"
    / "chest_xray"
    / "chest_xray"
)

TB_ROOT = (
    PROJECT_ROOT
    / "Datasets"
    / "raw"
    / "Tuberculosis"
    / "TB_Chest_Radiography_Database"
)

COMMON_ROOT = PROJECT_ROOT / "Datasets" / "common"

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

TRAIN_COUNT = 700
VAL_COUNT = 150
TEST_COUNT = 150

TOTAL_PER_CLASS = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

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
# HELPER FUNCTIONS
# ============================================================

def get_images(folder):
    """Return all image files inside a folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_folders():
    """Create common dataset directory structure."""

    for split in ["train", "val", "test"]:
        for class_name in ["Normal", "Pneumonia", "Tuberculosis"]:
            folder = COMMON_ROOT / split / class_name
            folder.mkdir(parents=True, exist_ok=True)


def split_images(images):
    """
    Randomly select and split images into:
    700 train, 150 validation, 150 test.
    """

    images = images.copy()
    random.shuffle(images)

    selected = images[:TOTAL_PER_CLASS]

    train = selected[:TRAIN_COUNT]
    val = selected[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test = selected[TRAIN_COUNT + VAL_COUNT:]

    return train, val, test


def copy_images(images, destination, prefix):
    """Copy images while preventing filename collisions."""

    destination.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(images, start=1):

        new_name = f"{prefix}_{index:04d}{source.suffix.lower()}"

        destination_path = destination / new_name

        shutil.copy2(source, destination_path)


def process_class(class_name, images, prefix):
    """Select 1000 images and create train/val/test split."""

    print("\n" + "=" * 60)
    print(f"PROCESSING: {class_name}")
    print("=" * 60)

    print(f"Available images: {len(images)}")

    if len(images) < TOTAL_PER_CLASS:
        raise ValueError(
            f"Not enough images for {class_name}. "
            f"Need {TOTAL_PER_CLASS}, found {len(images)}."
        )

    train, val, test = split_images(images)

    print(f"Selected images : {len(train) + len(val) + len(test)}")
    print(f"Train           : {len(train)}")
    print(f"Validation      : {len(val)}")
    print(f"Test            : {len(test)}")

    copy_images(
        train,
        COMMON_ROOT / "train" / class_name,
        f"{prefix}_train",
    )

    copy_images(
        val,
        COMMON_ROOT / "val" / class_name,
        f"{prefix}_val",
    )

    copy_images(
        test,
        COMMON_ROOT / "test" / class_name,
        f"{prefix}_test",
    )

    print(f"✓ {class_name} completed")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CHEXPERT COMMON DATASET CREATION")
    print("=" * 60)

    print("\nIMPORTANT:")
    print("Shenzhen dataset will NOT be used.")
    print("Original datasets will NOT be modified.")
    print("Images will be COPIED into Datasets/common/.")

    create_folders()

    # --------------------------------------------------------
    # PNEUMONIA
    # --------------------------------------------------------

    pneumonia_images = get_images(
        PNEUMONIA_ROOT / "train" / "PNEUMONIA"
    )

    # --------------------------------------------------------
    # TUBERCULOSIS
    # --------------------------------------------------------

    tb_images = get_images(
        TB_ROOT / "Tuberculosis"
    )

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------
    # Take 500 Normal images from Pneumonia dataset
    # and 500 Normal images from TB dataset.

    pneumonia_normal_images = get_images(
        PNEUMONIA_ROOT / "train" / "NORMAL"
    )

    tb_normal_images = get_images(
        TB_ROOT / "Normal"
    )

    if len(pneumonia_normal_images) < 500:
        raise ValueError(
            "Pneumonia NORMAL dataset contains fewer than 500 images."
        )

    if len(tb_normal_images) < 500:
        raise ValueError(
            "TB Normal dataset contains fewer than 500 images."
        )

    random.shuffle(pneumonia_normal_images)
    random.shuffle(tb_normal_images)

    normal_images = (
        pneumonia_normal_images[:500]
        + tb_normal_images[:500]
    )

    # --------------------------------------------------------
    # CREATE DATASET
    # --------------------------------------------------------

    process_class(
        "Normal",
        normal_images,
        "normal",
    )

    process_class(
        "Pneumonia",
        pneumonia_images,
        "pneumonia",
    )

    process_class(
        "Tuberculosis",
        tb_images,
        "tb",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COMMON DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print("\nNormal       : 1000")
    print("Pneumonia    : 1000")
    print("Tuberculosis : 1000")
    print("-----------------------")
    print("TOTAL        : 3000")

    print("\nTrain : 2100")
    print("Val   : 450")
    print("Test  : 450")

    print(f"\nLocation:")
    print(COMMON_ROOT)

    print("\nThis dataset is NOT used by the current training pipeline.")


if __name__ == "__main__":
    main()