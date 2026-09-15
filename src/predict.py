import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import (
    DEVICE,
    IMAGE_SIZE,
    PNEUMONIA_MODEL_PATH,
    TB_MODEL_PATH,
)

from model import create_model


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_model(model_path):

    model = create_model()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ---------------------------------------------------------
# Predict single image
# ---------------------------------------------------------

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move image to GPU
    image_tensor = image_tensor.to(DEVICE)

    # Load models
    pneumonia_model = load_model(
        PNEUMONIA_MODEL_PATH
    )

    tb_model = load_model(
        TB_MODEL_PATH
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    with torch.no_grad():

        pneumonia_output = pneumonia_model(
            image_tensor
        )

        tb_output = tb_model(
            image_tensor
        )

        # Convert logits to probabilities
        pneumonia_probabilities = torch.softmax(
            pneumonia_output,
            dim=1
        )

        tb_probabilities = torch.softmax(
            tb_output,
            dim=1
        )

    # -----------------------------------------------------
    # Extract probabilities
    # -----------------------------------------------------

    pneumonia_normal_probability = (
        pneumonia_probabilities[0][0].item()
    )

    pneumonia_probability = (
        pneumonia_probabilities[0][1].item()
    )

    tb_normal_probability = (
        tb_probabilities[0][0].item()
    )

    tb_probability = (
        tb_probabilities[0][1].item()
    )

    # -----------------------------------------------------
    # Predicted classes
    # -----------------------------------------------------

    pneumonia_prediction = (
        "PNEUMONIA"
        if pneumonia_probability >= 0.5
        else "NORMAL"
    )

    tb_prediction = (
        "TUBERCULOSIS"
        if tb_probability >= 0.5
        else "NORMAL"
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("CHEXPERT X-RAY PREDICTION")
    print("=" * 60)

    print("Image:", image_path)
    print("Device:", DEVICE)

    print("\nPNEUMONIA")
    print("-" * 30)
    print(
        f"Normal probability    : "
        f"{pneumonia_normal_probability * 100:.2f}%"
    )
    print(
        f"Pneumonia probability : "
        f"{pneumonia_probability * 100:.2f}%"
    )
    print(
        f"Prediction            : "
        f"{pneumonia_prediction}"
    )

    print("\nTUBERCULOSIS")
    print("-" * 30)
    print(
        f"Normal probability    : "
        f"{tb_normal_probability * 100:.2f}%"
    )
    print(
        f"TB probability        : "
        f"{tb_probability * 100:.2f}%"
    )
    print(
        f"Prediction            : "
        f"{tb_prediction}"
    )

    print("\n" + "=" * 60)

    # -----------------------------------------------------
    # Return structured result
    # -----------------------------------------------------

    return {
        "image": str(image_path),

        "pneumonia": {
            "probability": pneumonia_probability,
            "normal_probability": pneumonia_normal_probability,
            "prediction": pneumonia_prediction,
        },

        "tuberculosis": {
            "probability": tb_probability,
            "normal_probability": tb_normal_probability,
            "prediction": tb_prediction,
        },
    }


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python src\\predict.py "
            "\"path\\to\\xray.jpg\""
        )

        sys.exit(1)

    image_path = sys.argv[1]

    predict_image(image_path)