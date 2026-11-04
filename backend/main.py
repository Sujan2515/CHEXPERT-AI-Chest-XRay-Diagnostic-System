
from contextlib import asynccontextmanager
from pathlib import Path
from io import BytesIO

import torch
import torch.nn.functional as F

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

from src.config import (
    DEVICE,
    IMAGE_SIZE,
    PNEUMONIA_MODEL_PATH,
    TB_MODEL_PATH
)
from src.model import create_model


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

def load_checkpoint(model_path):

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = create_model()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# APPLICATION STARTUP
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading CHEXPERT models...")

    app.state.pneumonia_model = load_checkpoint(
        PNEUMONIA_MODEL_PATH
    )

    app.state.tb_model = load_checkpoint(
        TB_MODEL_PATH
    )

    print("Both models loaded successfully!")

    yield

    app.state.pneumonia_model = None
    app.state.tb_model = None


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="CHEXPERT API",
    description="Chest X-ray research prediction API",
    version="1.0.0",
    lifespan=lifespan
)



# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://chexpert-ai.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "CHEXPERT API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "pneumonia_model_loaded":
            app.state.pneumonia_model is not None,
        "tb_model_loaded":
            app.state.tb_model is not None
    }


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_image(model, image_tensor, class_names):

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index].item()) * 100,
            2
        ),
        "probabilities": {
            class_names[i]: round(
                float(probabilities[i].item()) * 100,
                2
            )
            for i in range(len(class_names))
        }
    }


# --------------------------------------------------
# CHEST X-RAY PREDICTION API
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB."
        )

    try:

        image = Image.open(
            BytesIO(contents)
        )

        image.verify()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    pneumonia_result = predict_image(
        app.state.pneumonia_model,
        image_tensor,
        ["Normal", "Pneumonia"]
    )

    tb_result = predict_image(
        app.state.tb_model,
        image_tensor,
        ["Normal", "Tuberculosis"]
    )

    return {
        "status": "success",
        "filename": file.filename,
        "results": {
            "pneumonia": pneumonia_result,
            "tuberculosis": tb_result
        },
        "notice": (
            "Experimental research predictions only. "
            "This system is not a medical diagnosis."
        )
    }
