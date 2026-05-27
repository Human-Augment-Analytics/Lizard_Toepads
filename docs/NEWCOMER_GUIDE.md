# Newcomer Guide: Lizard Toepads & LizardMorph

Welcome to the Lizard Morphometrics project! This guide explains the relationship between the two main repositories and how to get started with the pipeline.

## System Overview

The project is split into two distinct parts:

1.  **Lizard Toepads (this repository)**: The Machine Learning Research & Training core.
    *   **Purpose**: Handles data preprocessing (TPS to YOLO), training the YOLOv11 detectors (Stage 1), and the landmark prediction research (Stage 2 / ml-morph).
    *   **Stack**: Python, Ultralytics YOLOv11, dlib, PyTorch.
    *   **Primary Environment**: PACE ICE Cluster (GPU-intensive training).

2.  **LizardMorph (Web App)**: The User Interface & Deployment platform.
    *   **Purpose**: A web application that allows researchers to upload images, run inference using the models trained in "Lizard Toepads", and visualize results.
    *   **Stack**: Backend (Python/FastAPI), Frontend (Next.js/React).
    *   **Repo Location**: Typically located at `../LizardMorph` on dev machines.

---

## 1. Lizard Toepads: The ML Pipeline

The pipeline follows a two-stage architecture:

### Stage 1: Detection (YOLOv11)
*   **Goal**: Detect regions of interest (ROI) such as Fingers, Toes, Rulers, and IDs.
*   **Key Scripts**:
    *   `scripts/preprocessing/generate_bottom_view_labels.py`: Converts TPS landmarks to YOLO labels.
    *   `scripts/training/train_yolo.py`: Trains the detection model.
*   **Config**: `configs/H4.yaml` (Standard bilateral detection config).

### Stage 2: Landmark Prediction (ml-morph)
*   **Goal**: Predict precise (x, y) coordinates within the detected ROIs.
*   **Location**: `ml-morph/` directory.
*   **Key Scripts**:
    *   `ml-morph/shape_trainer.py`: Trains the dlib shape predictor.
    *   `ml-morph/scripts/preprocessing/generate_yolo_bbox_xml.py`: Connects Stage 1 and Stage 2 by generating training data using YOLO bounding boxes.

---

## 2. LizardMorph: The Web Application

LizardMorph provides the production environment for the models.

*   **Deployment**: Models trained in `Lizard Toepads` (e.g., `best.pt` for YOLO, `predictor.dat` for dlib) are moved to the `backend/models/` directory of the LizardMorph app.
*   **Inference**: The web app runs a specialized inference script (`backend/utils.py`) that handles image uploads and sequential model execution.
*   **Rotation Sensitivity**: Note that the dlib predictors are sensitive to orientation. Research in `Lizard Toepads` (see `docs/Experiment_PCA_Summary.md`) informs how the web app handles image rotation before prediction.

---

## Getting Started

### For ML Researchers (Lizard Toepads)
1.  **Environment**: Use `uv` to manage dependencies.
    ```bash
    uv sync
    # Install PyTorch with CUDA for PACE
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```
2.  **Training**: Follow the step-by-step instructions in `docs/RUN_EXPERIMENTS_STEP_BY_STEP.md`.
3.  **Branching**: Most active development happens on the `leyang/ml-morph` branch.

### For Web Developers (LizardMorph)
1.  Navigate to the `LizardMorph` repository.
2.  Follow the setup instructions in its own README for the FastAPI backend and Next.js frontend.
3.  Ensure you have the latest models from this repository placed in the appropriate `models/` folder.

---

## Contact & Documentation
*   **Codebase Overview**: `docs/Codebase_Overview.md`
*   **PACE Experiments**: `docs/ML_MORPH_PACE_EXPERIMENTS.md`
*   **Model Results**: `docs/COMPARISON_BASELINE_VS_OBB.md`
