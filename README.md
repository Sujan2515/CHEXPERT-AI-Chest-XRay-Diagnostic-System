# CHEXPERT — AI-Based Chest X-Ray Diagnostic System

An AI-based chest X-ray diagnostic system designed to assist in the detection of respiratory diseases, initially focusing on **Pneumonia** and **Tuberculosis**. The system uses deep learning for X-ray classification and is designed to be extended with Explainable AI, Agentic AI, and multilingual report generation.

---

## 📌 Project Overview

CHEXPERT is a major project focused on developing an AI-assisted chest X-ray diagnostic system for healthcare support, particularly in rural and resource-constrained settings.

The current implementation uses **DenseNet-121**, a convolutional neural network (CNN), with transfer learning to classify chest X-ray images.

Two separate binary classification models are currently implemented:

- **Pneumonia Detection**
- **Tuberculosis Detection**

The planned complete system will additionally incorporate:

- Explainable AI (XAI)
- Agentic AI for interactive diagnostic reasoning
- Generative AI for report generation
- Multilingual patient-friendly explanations
- Text-to-Speech (TTS)
- Interactive diagnostic interface

> **Note:** The current repository primarily contains the deep-learning classification pipeline. The Agentic AI and Generative AI components are planned extensions of the system.

---

## 🏗️ Current System Architecture

```text
                 Chest X-Ray
                      │
                      ▼
             Image Preprocessing
                      │
                      ▼
              DenseNet-121 CNN
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Pneumonia Model           TB Model
          │                       │
          ▼                       ▼
 Normal / Pneumonia        Normal / TB
          │                       │
          └───────────┬───────────┘
                      ▼
                Prediction