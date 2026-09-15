from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config import (
    PNEUMONIA_ROOT,
    TB_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# CUSTOM BINARY DATASET
# ============================================================

class BinaryImageDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label = self.labels[index]

        image = datasets.folder.default_loader(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_paths(folder):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    folder = Path(folder)

    paths = []

    if not folder.exists():
        return paths

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(str(path))

    return sorted(paths)


# ============================================================
# PNEUMONIA DATASET
#
# Pneumonia model:
#
# 1 = Pneumonia
# 0 = Normal OR TB
#
# IMPORTANT:
# Official Pneumonia TEST remains completely untouched.
# TB data is split first so TB test images never enter training.
# ============================================================

def get_pneumonia_datasets():

    pneumonia_train_root = PNEUMONIA_ROOT / "train"
    pneumonia_test_root = PNEUMONIA_ROOT / "test"

    tb_normal_root = TB_ROOT / "Normal"
    tb_positive_root = TB_ROOT / "Tuberculosis"

    # --------------------------------------------------------
    # Pneumonia training data
    # --------------------------------------------------------

    pneumonia_positive = get_image_paths(
        pneumonia_train_root / "PNEUMONIA"
    )

    pneumonia_normal = get_image_paths(
        pneumonia_train_root / "NORMAL"
    )

    # --------------------------------------------------------
    # Official Pneumonia test data
    # NEVER used for training
    # --------------------------------------------------------

    pneumonia_test_normal = get_image_paths(
        pneumonia_test_root / "NORMAL"
    )

    pneumonia_test_positive = get_image_paths(
        pneumonia_test_root / "PNEUMONIA"
    )

    # --------------------------------------------------------
    # Split TB dataset
    #
    # 70% train
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    tb_normal = get_image_paths(tb_normal_root)
    tb_positive = get_image_paths(tb_positive_root)

    tb_normal_train, tb_normal_temp = train_test_split(
        tb_normal,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    tb_normal_val, tb_normal_test = train_test_split(
        tb_normal_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    tb_positive_train, tb_positive_temp = train_test_split(
        tb_positive,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    tb_positive_val, tb_positive_test = train_test_split(
        tb_positive_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # TRAIN
    #
    # Positive = Pneumonia
    # Negative = Normal + TB
    # --------------------------------------------------------

    train_paths = (
        pneumonia_positive
        + pneumonia_normal
        + tb_normal_train
        + tb_positive_train
    )

    train_labels = (
        [1] * len(pneumonia_positive)
        + [0] * len(pneumonia_normal)
        + [0] * len(tb_normal_train)
        + [0] * len(tb_positive_train)
    )

    # --------------------------------------------------------
    # Split training data into train / validation
    # --------------------------------------------------------

    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            train_paths,
            train_labels,
            test_size=0.10,
            random_state=RANDOM_SEED,
            stratify=train_labels,
        )
    )

    train_dataset = BinaryImageDataset(
        train_paths,
        train_labels,
        transform=train_transform,
    )

    val_dataset = BinaryImageDataset(
        val_paths,
        val_labels,
        transform=eval_transform,
    )

    # --------------------------------------------------------
    # TEST
    #
    # Pneumonia official test
    # +
    # TB held-out test
    # --------------------------------------------------------

    test_paths = (
        pneumonia_test_normal
        + pneumonia_test_positive
        + tb_normal_test
        + tb_positive_test
    )

    test_labels = (
        [0] * len(pneumonia_test_normal)
        + [1] * len(pneumonia_test_positive)
        + [0] * len(tb_normal_test)
        + [0] * len(tb_positive_test)
    )

    test_dataset = BinaryImageDataset(
        test_paths,
        test_labels,
        transform=eval_transform,
    )

    print("\nPNEUMONIA MODEL DATASET")
    print("-" * 50)

    print(
        "Pneumonia positive:",
        len(pneumonia_positive)
    )

    print(
        "Pneumonia normal:",
        len(pneumonia_normal)
    )

    print(
        "TB train negatives:",
        len(tb_normal_train) + len(tb_positive_train)
    )

    print(
        "TB validation:",
        len(tb_normal_val) + len(tb_positive_val)
    )

    print(
        "TB test negatives:",
        len(tb_normal_test) + len(tb_positive_test)
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# TB DATASET
#
# TB model:
#
# 1 = Tuberculosis
# 0 = Normal OR Pneumonia
#
# Again, all test images remain completely separate.
# ============================================================

def get_tb_datasets():

    pneumonia_train_root = PNEUMONIA_ROOT / "train"
    pneumonia_test_root = PNEUMONIA_ROOT / "test"

    tb_normal_root = TB_ROOT / "Normal"
    tb_positive_root = TB_ROOT / "Tuberculosis"

    # --------------------------------------------------------
    # TB data
    # --------------------------------------------------------

    tb_normal = get_image_paths(tb_normal_root)
    tb_positive = get_image_paths(tb_positive_root)

    # --------------------------------------------------------
    # Pneumonia training data
    # --------------------------------------------------------

    pneumonia_positive = get_image_paths(
        pneumonia_train_root / "PNEUMONIA"
    )

    pneumonia_normal = get_image_paths(
        pneumonia_train_root / "NORMAL"
    )

    # --------------------------------------------------------
    # Official Pneumonia test data
    # --------------------------------------------------------

    pneumonia_test_normal = get_image_paths(
        pneumonia_test_root / "NORMAL"
    )

    pneumonia_test_positive = get_image_paths(
        pneumonia_test_root / "PNEUMONIA"
    )

    # --------------------------------------------------------
    # Split TB data
    # --------------------------------------------------------

    tb_normal_train, tb_normal_temp = train_test_split(
        tb_normal,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    tb_normal_val, tb_normal_test = train_test_split(
        tb_normal_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    tb_positive_train, tb_positive_temp = train_test_split(
        tb_positive,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    tb_positive_val, tb_positive_test = train_test_split(
        tb_positive_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # Build negative pool
    #
    # Negative = Normal + Pneumonia
    # --------------------------------------------------------

    negative_paths = (
        tb_normal_train
        + pneumonia_positive
        + pneumonia_normal
    )

    negative_labels = [0] * len(negative_paths)

    # Positive = TB
    positive_paths = tb_positive_train
    positive_labels = [1] * len(positive_paths)

    # --------------------------------------------------------
    # Balance the dataset
    #
    # Use 2 negative images for every TB image.
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    target_negative_count = min(
        len(negative_paths),
        len(positive_paths) * 2
    )

    selected_negative_indices = random.sample(
        range(len(negative_paths)),
        target_negative_count
    )

    selected_negative_paths = [
        negative_paths[i]
        for i in selected_negative_indices
    ]

    selected_negative_labels = [
        0
    ] * len(selected_negative_paths)

    balanced_paths = (
        positive_paths
        + selected_negative_paths
    )

    balanced_labels = (
        positive_labels
        + selected_negative_labels
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION
    # --------------------------------------------------------

    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            balanced_paths,
            balanced_labels,
            test_size=0.15,
            random_state=RANDOM_SEED,
            stratify=balanced_labels,
        )
    )

    train_dataset = BinaryImageDataset(
        train_paths,
        train_labels,
        transform=train_transform,
    )

    val_dataset = BinaryImageDataset(
        val_paths,
        val_labels,
        transform=eval_transform,
    )

    # --------------------------------------------------------
    # TEST
    #
    # TB held-out test
    # +
    # Pneumonia official test
    # --------------------------------------------------------

    test_paths = (
        tb_normal_test
        + tb_positive_test
        + pneumonia_test_normal
        + pneumonia_test_positive
    )

    test_labels = (
        [0] * len(tb_normal_test)
        + [1] * len(tb_positive_test)
        + [0] * len(pneumonia_test_normal)
        + [0] * len(pneumonia_test_positive)
    )

    test_dataset = BinaryImageDataset(
        test_paths,
        test_labels,
        transform=eval_transform,
    )

    print("\nTB MODEL DATASET")
    print("-" * 50)

    print(
        "TB positive training:",
        len(tb_positive_train)
    )

    print(
        "TB normal training:",
        len(tb_normal_train)
    )

    print(
        "Pneumonia negatives:",
        len(pneumonia_positive) + len(pneumonia_normal)
    )

    print(
        "TB validation:",
        len(tb_normal_val) + len(tb_positive_val)
    )

    print(
        "TB test:",
        len(tb_normal_test) + len(tb_positive_test)
    )

    print(
        "Balanced training images:",
        len(balanced_paths)
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# DATALOADER
# ============================================================

def create_loader(dataset, shuffle=False):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CHEXPERT LEAKAGE-FREE DATASET TEST")
    print("=" * 60)

    pneumonia_train, pneumonia_val, pneumonia_test = (
        get_pneumonia_datasets()
    )

    tb_train, tb_val, tb_test = (
        get_tb_datasets()
    )

    print("\n" + "=" * 60)
    print("DATASET CONSTRUCTION SUCCESSFUL")
    print("=" * 60)