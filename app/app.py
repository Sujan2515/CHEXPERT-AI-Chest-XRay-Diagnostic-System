import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
import sys


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CHEXPERT - Chest X-Ray Diagnostic System",
    page_icon="🩻",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

MODEL_DIR = PROJECT_ROOT / "models"

PNEUMONIA_MODEL_PATH = MODEL_DIR / "pneumonia_model.pth"
TB_MODEL_PATH = MODEL_DIR / "tb_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# Same preprocessing used by our trained models
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
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


# ============================================================
# LOAD BOTH TRAINED MODELS
# ============================================================

@st.cache_resource
def load_all_models():

    if not PNEUMONIA_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pneumonia model not found:\n"
            f"{PNEUMONIA_MODEL_PATH}"
        )

    if not TB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tuberculosis model not found:\n"
            f"{TB_MODEL_PATH}"
        )

    pneumonia_model = load_model(
        PNEUMONIA_MODEL_PATH
    )

    tb_model = load_model(
        TB_MODEL_PATH
    )

    return pneumonia_model, tb_model


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_xray(image, pneumonia_model, tb_model):

    # Convert image to RGB
    image = image.convert("RGB")

    # Apply same preprocessing as predict.py
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move image to GPU / CPU
    image_tensor = image_tensor.to(DEVICE)

    # Disable gradients during inference
    with torch.no_grad():

        # --------------------------------------------
        # PNEUMONIA MODEL
        # --------------------------------------------

        pneumonia_output = pneumonia_model(
            image_tensor
        )

        pneumonia_probabilities = torch.softmax(
            pneumonia_output,
            dim=1
        )

        # --------------------------------------------
        # TB MODEL
        # --------------------------------------------

        tb_output = tb_model(
            image_tensor
        )

        tb_probabilities = torch.softmax(
            tb_output,
            dim=1
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

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

    # ========================================================
    # PREDICTIONS
    # ========================================================

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

    return {
        "pneumonia_prediction": pneumonia_prediction,
        "pneumonia_probability": pneumonia_probability,
        "pneumonia_normal_probability":
            pneumonia_normal_probability,

        "tb_prediction": tb_prediction,
        "tb_probability": tb_probability,
        "tb_normal_probability":
            tb_normal_probability
    }


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        color: #16324F;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #5f6b7a;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        color: #16324F;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">CHEXPERT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Chest X-Ray Diagnostic System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Deep-learning based detection of Pneumonia "
    "and Tuberculosis from chest X-ray images."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("CHEXPERT")

    st.write(
        "AI-assisted chest X-ray analysis "
        "using DenseNet-121."
    )

    st.divider()

    st.subheader("Current Modules")

    st.write("🩻 X-Ray Upload")
    st.write("🧠 DenseNet-121")
    st.write("🫁 Pneumonia Detection")
    st.write("🦠 TB Detection")
    st.write("📄 Basic Report")

    st.divider()

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    st.divider()

    st.subheader("Planned Modules")

    st.write("🔍 Explainable AI")
    st.write("🤖 Agentic AI")
    st.write("🌐 Multilingual Reports")
    st.write("🔊 Text-to-Speech")

    st.divider()

    st.caption(
        "Major Project — 7th Semester"
    )


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

try:

    pneumonia_model, tb_model = load_all_models()

except Exception as e:

    st.error(
        "Unable to load the trained models."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Upload Chest X-Ray'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # --------------------------------------------------------
    # IMAGE + PATIENT INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Uploaded X-Ray")

        st.image(
            image,
            caption="Chest X-Ray",
            use_container_width=True
        )

    with col2:

        st.subheader("Patient Information")

        patient_id = st.text_input(
            "Patient ID",
            placeholder="Enter patient ID"
        )

        patient_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

        patient_gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

    st.divider()


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze X-Ray",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing X-ray using DenseNet-121..."
        ):

            results = predict_xray(
                image,
                pneumonia_model,
                tb_model
            )

        st.success(
            "X-ray analysis completed."
        )


        # ====================================================
        # EXTRACT RESULTS
        # ====================================================

        pneumonia_prediction = results[
            "pneumonia_prediction"
        ]

        pneumonia_probability = results[
            "pneumonia_probability"
        ]

        pneumonia_normal_probability = results[
            "pneumonia_normal_probability"
        ]

        tb_prediction = results[
            "tb_prediction"
        ]

        tb_probability = results[
            "tb_probability"
        ]

        tb_normal_probability = results[
            "tb_normal_probability"
        ]


        # ====================================================
        # DIAGNOSTIC RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Diagnostic Results'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # PNEUMONIA RESULT
        # ----------------------------------------------------

        with col1:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.subheader("🫁 Pneumonia")

            st.write("Prediction")

            st.markdown(
                f"## {pneumonia_prediction}"
            )

            st.progress(
                int(
                    pneumonia_probability * 100
                )
            )

            st.write(
                f"**Pneumonia probability:** "
                f"{pneumonia_probability * 100:.2f}%"
            )

            st.write(
                f"**Normal probability:** "
                f"{pneumonia_normal_probability * 100:.2f}%"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # TB RESULT
        # ----------------------------------------------------

        with col2:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.subheader("🦠 Tuberculosis")

            st.write("Prediction")

            st.markdown(
                f"## {tb_prediction}"
            )

            st.progress(
                int(
                    tb_probability * 100
                )
            )

            st.write(
                f"**TB probability:** "
                f"{tb_probability * 100:.2f}%"
            )

            st.write(
                f"**Normal probability:** "
                f"{tb_normal_probability * 100:.2f}%"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # OVERALL RESULT
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            'Overall Assessment'
            '</div>',
            unsafe_allow_html=True
        )

        if (
            pneumonia_prediction == "PNEUMONIA"
            and tb_prediction == "TUBERCULOSIS"
        ):

            st.warning(
                "The models detected both Pneumonia "
                "and Tuberculosis."
            )

        elif pneumonia_prediction == "PNEUMONIA":

            st.warning(
                "The Pneumonia classifier detected "
                "Pneumonia."
            )

        elif tb_prediction == "TUBERCULOSIS":

            st.warning(
                "The Tuberculosis classifier detected "
                "Tuberculosis."
            )

        else:

            st.success(
                "Neither Pneumonia nor Tuberculosis "
                "was detected by the classifiers."
            )


        # ====================================================
        # BASIC REPORT
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            'Diagnostic Report'
            '</div>',
            unsafe_allow_html=True
        )

        report = f"""
CHEXPERT
AI-BASED CHEST X-RAY DIAGNOSTIC SYSTEM
========================================

PATIENT INFORMATION
-------------------
Patient ID: {patient_id if patient_id else "Not provided"}
Age: {patient_age}
Gender: {patient_gender}


PNEUMONIA ASSESSMENT
--------------------
Prediction: {pneumonia_prediction}
Pneumonia Probability: {pneumonia_probability * 100:.2f}%
Normal Probability: {pneumonia_normal_probability * 100:.2f}%


TUBERCULOSIS ASSESSMENT
-----------------------
Prediction: {tb_prediction}
TB Probability: {tb_probability * 100:.2f}%
Normal Probability: {tb_normal_probability * 100:.2f}%


MODEL INFORMATION
-----------------
Architecture: DenseNet-121
Framework: PyTorch
Input Size: 224 x 224
Device: {DEVICE}


DISCLAIMER
----------
This system is an academic/research prototype.
The predictions are intended for decision-support
purposes and should not replace evaluation by a
qualified healthcare professional.
"""


        st.text_area(
            "Report",
            report,
            height=450
        )


        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        st.download_button(
            "📄 Download Report",
            report,
            file_name="CHEXPERT_Report.txt",
            mime="text/plain",
            use_container_width=True
        )


else:

    st.info(
        "Upload a chest X-ray image above to begin analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CHEXPERT | DenseNet-121 | PyTorch | "
    "Academic Project Prototype"
)