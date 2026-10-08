from pathlib import Path
import shutil
import random

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original datasets
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

# New common dataset
COMMON_ROOT = PROJECT_ROOT / "Datasets" / "common"

# Number of images required
PNEUMONIA_COUNT = 1000
TB_COUNT = 700

RANDOM_SEED = 42

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_images(folder):
    """Return all image files inside a folder."""
    return [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# CREATE PNEUMONIA DATASET
# ============================================================

def create_pneumonia_dataset():

    source_folder = PNEUMONIA_ROOT / "train" / "PNEUMONIA"
    destination_folder = COMMON_ROOT / "Pneumonia"

    destination_folder.mkdir(parents=True, exist_ok=True)

    # Remove previously copied Pneumonia images
    for file in destination_folder.iterdir():
        if file.is_file():
            file.unlink()

    images = get_images(source_folder)

    print("\nPNEUMONIA")
    print("-" * 50)
    print("Available images:", len(images))

    if len(images) < PNEUMONIA_COUNT:
        raise ValueError(
            f"Only {len(images)} Pneumonia images available, "
            f"but {PNEUMONIA_COUNT} are required."
        )

    # Reproducible random selection
    random.seed(RANDOM_SEED)
    selected_images = random.sample(images, PNEUMONIA_COUNT)

    # Copy images
    for index, image_path in enumerate(selected_images, start=1):

        extension = image_path.suffix.lower()

        new_name = f"pneumonia_{index:04d}{extension}"

        destination = destination_folder / new_name

        shutil.copy2(image_path, destination)

    print("Copied images:", PNEUMONIA_COUNT)
    print("Destination:", destination_folder)


# ============================================================
# CREATE TB DATASET
# ============================================================

def create_tb_dataset():

    source_folder = TB_ROOT / "Tuberculosis"
    destination_folder = COMMON_ROOT / "Tuberculosis"

    destination_folder.mkdir(parents=True, exist_ok=True)

    # Remove previously copied TB images
    for file in destination_folder.iterdir():
        if file.is_file():
            file.unlink()

    images = get_images(source_folder)

    print("\nTUBERCULOSIS")
    print("-" * 50)
    print("Available images:", len(images))

    if len(images) < TB_COUNT:
        raise ValueError(
            f"Only {len(images)} TB images available, "
            f"but {TB_COUNT} are required."
        )

    # We want all 700 TB images
    images = sorted(images)

    selected_images = images[:TB_COUNT]

    # Copy images
    for index, image_path in enumerate(selected_images, start=1):

        extension = image_path.suffix.lower()

        new_name = f"tb_{index:04d}{extension}"

        destination = destination_folder / new_name

        shutil.copy2(image_path, destination)

    print("Copied images:", TB_COUNT)
    print("Destination:", destination_folder)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CREATING COMMON CHEST X-RAY DATASET")
    print("=" * 60)

    create_pneumonia_dataset()
    create_tb_dataset()

    print("\n" + "=" * 60)
    print("COMMON DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nPneumonia : {PNEUMONIA_COUNT} images")
    print(f"Tuberculosis: {TB_COUNT} images")
    print(f"Total     : {PNEUMONIA_COUNT + TB_COUNT} images")

    print(f"\nLocation:")
    print(COMMON_ROOT)