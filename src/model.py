import torch
import torch.nn as nn
from torchvision import models


def create_model():

    # Load pretrained DenseNet121
    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    # Get number of input features
    num_features = model.classifier.in_features

    # Replace the original classifier
    model.classifier = nn.Linear(
        num_features,
        2
    )

    return model


def create_loss_function(dataset):
    """
    Create a weighted CrossEntropyLoss based on
    the class distribution of the training dataset.
    """

    # Custom BinaryImageDataset
    if hasattr(dataset, "labels"):
        labels = dataset.labels

    # PyTorch Subset
    elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):

        base_dataset = dataset.dataset

        if hasattr(base_dataset, "labels"):
            labels = [
                base_dataset.labels[i]
                for i in dataset.indices
            ]

        elif hasattr(base_dataset, "targets"):
            labels = [
                base_dataset.targets[i]
                for i in dataset.indices
            ]

        else:
            raise AttributeError(
                "Dataset does not contain labels or targets."
            )

    # torchvision ImageFolder
    elif hasattr(dataset, "targets"):
        labels = dataset.targets

    else:
        raise AttributeError(
            "Dataset does not contain labels or targets."
        )

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)

    # Count samples in each class
    class_counts = torch.bincount(labels, minlength=2)

    # Inverse-frequency weighting
    weights = 1.0 / class_counts.float()

    # Normalize weights
    weights = (
        weights / weights.sum()
    ) * len(class_counts)

    return nn.CrossEntropyLoss(
        weight=weights
    )

if __name__ == "__main__":

    print("=" * 60)
    print("CHEXPERT MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = create_model()

    model = model.to(device)

    print("Model: DenseNet121")
    print("Output classes: 2")

    # Test with one batch of fake images
    test_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    with torch.no_grad():
        output = model(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)

    print("\nModel successfully loaded.")