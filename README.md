# AI-Powered Image Quality & Defect Detection Platform

[![CI/CD Pipeline](https://github.com/lakshminarasimmann/mage-quality-defect-detection/actions/workflows/main.yml/badge.svg)](https://github.com/lakshminarasimmann/mage-quality-defect-detection/actions/workflows/main.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-5C3EE8?logo=opencv)](https://opencv.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)

A complete, production-ready AI full-stack application that accepts images and automatically evaluates visual quality, identifies specific photographic defects, localizes spatial degradations using thermal heatmaps, and persists historical diagnostics.

---

## 📑 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [End-to-End System Workflow](#-end-to-end-system-workflow)
3. [Technology Stack](#-technology-stack)
4. [Theoretical Methodology: Computer Vision & Feature Extraction](#-theoretical-methodology-computer-vision--feature-extraction)
5. [AI / Machine Learning Model Architecture](#-ai--machine-learning-model-architecture)
6. [Dataset & Synthetic Degradation Pipeline](#-dataset--synthetic-degradation-pipeline)
7. [Explainability & Spatial Heatmaps (Localization)](#-explainability--spatial-heatmaps-localization)
8. [Codebase Architecture & File-by-File Guide](#-codebase-architecture--file-by-file-guide)
9. [REST API Specification](#-rest-api-specification)
10. [Quick Start & Setup Guide](#-quick-start--setup-guide)
11. [Command-Line Interface (CLI) Tool](#-command-line-interface-cli-tool)
12. [Automated Testing & Verification](#-automated-testing--verification)

---

## 📌 Project Overview & Problem Statement

In real-world computer vision applications, incoming images often suffer from various physical and digital degradations—such as optical defocus blur, sensor noise, severe underexposure/overexposure, and data transmission corruption. 

This platform provides an automated quality evaluation system that determines whether an uploaded image is **Acceptable**, **Degraded**, or **Defective** without relying on external cloud vision APIs.

### Core Objectives:
* **Zero External API Costs**: Runs entirely on local open-source libraries (PyTorch and OpenCV).
* **Hybrid Intelligence**: Combines deterministic computer vision feature extraction with a multi-task neural network to avoid "black box" decisions.
* **Spatial Explainability**: Generates visual degradation heatmaps pinpointing exact regions of blur or corruption.
* **Batch Processing**: Supports concurrent multi-image triage via an interactive drag-and-drop interface.
* **Transactional Persistence**: Records historical diagnostics in an SQLite database.

---

## 🔄 End-to-End System Workflow

```
[User Browser]
       │
       ▼ (1. Drag-and-Drop Image / Batch of Images)
[React 18 SPA Frontend] ──(2. Multipart HTTP POST Request)──▶ [FastAPI Backend]
                                                                      │
                                   ┌──────────────────────────────────┴──────────────────────────────────┐
                                   ▼                                                                     ▼
                      [Computer Vision Engine (OpenCV)]                                      [Database Layer (SQLite)]
                      - Sharpness via Laplacian Variance                                     - Persists raw scores,
                      - Exposure via HSV Luminance                                             defect categories, and
                      - Noise via High-Frequency Residuals                                     timestamps using SQLAlchemy
                      - Dynamic Range Clipping Index
                      - 16x16 Grid Spatial Heatmap Generator
                                   │
                                   ▼ (Feature Vector [6-D])
                      [PyTorch Multi-Task MLP Network]
                      - Regresses Overall Quality Score (0 - 100)
                      - Predicts Defect Probabilities (Blur, Noise, Exposure, Corruption)
                                   │
                                   ▼
                      [JSON Response + Base64 Heatmap Overlay] ──▶ [React Frontend Visualizer]
```

### Execution Steps:
1. **User Interaction**: The user uploads single or multiple images through the React user interface.
2. **Preprocessing & Feature Extraction**: The FastAPI backend receives the binary stream and resizes images to a standard dimension ($512 \times 512$). OpenCV calculates statistical metrics representing sharpness, luminance, contrast, noise, saturation, and corruption.
3. **Heatmap Generation**: OpenCV divides the image into a $16 \times 16$ spatial grid, evaluates localized edge transitions, and renders a JET-colormap heatmap overlay.
4. **Neural Inference**: The 6 extracted statistical features are fed into a trained PyTorch neural network that outputs the overall continuous quality score and defect probabilities.
5. **Persistence**: The results are stored in SQLite.
6. **Visualization**: The frontend renders the quality score meter, severity tags, statistical breakdown bars, and the side-by-side thermal heatmap.

---

## 🛠 Technology Stack

| Layer | Technologies | Role & Purpose |
| :--- | :--- | :--- |
| **Frontend** | React 18, Vite, Vanilla CSS | Single Page Application with dark-mode glassmorphic aesthetics, drag-and-drop dropzone, dynamic radial gauge, and heatmap visualizer. |
| **Backend API** | Python 3.11, FastAPI, Uvicorn | Asynchronous REST gateway handling file parsing, validation, ML pipeline orchestration, and OpenAPI documentation. |
| **Machine Learning** | PyTorch (`torch`, `torchvision`) | Multi-task Multilayer Perceptron (MLP) neural network with dual output heads. |
| **Computer Vision** | OpenCV (`opencv-python-headless`), NumPy | Deterministic photometric feature extraction, edge filtering, and spatial heatmap generation. |
| **Database** | SQLite, SQLAlchemy 2.0 ORM | Lightweight, zero-config relational database for analysis logging. |
| **DevOps & CI/CD** | GitHub Actions, Docker, Docker Compose | Continuous integration pipeline executing automated test suites and multi-stage container builds. |
| **Testing** | Pytest, HTTPX | Automated unit and integration testing suite for API endpoints and ML modules. |

---

## 🔬 Theoretical Methodology: Computer Vision & Feature Extraction

Rather than treating image quality assessment as an uninterpretable deep neural network problem, our solution extracts domain-engineered computer vision features:

### 1. Blur / Sharpness Analysis (Laplacian Variance)
* **Theory**: In an in-focus, sharp image, edges are characterized by rapid transitions in pixel intensity (high spatial frequencies). Blurring acts as a low-pass filter that smooths out these rapid transitions.
* **Mechanism**: We convolve the grayscale image with a discrete Laplacian second-derivative operator. Sharp images generate significant variance across the edge response map, whereas blurry images yield near-zero variance.

### 2. Exposure & Photometric Distribution (HSV Luminance)
* **Theory**: Converting RGB images to the HSV (Hue, Saturation, Value) color space separates chromatic information from intensity.
* **Mechanism**: 
  * The **Mean of the Value channel** represents average scene luminance. Extremely low mean values indicate underexposure (crushed shadows), while high values indicate overexposure (blown-out highlights).
  * The **Standard Deviation of the Value channel** reflects overall scene dynamic contrast.

### 3. Image Noise Estimation (Residual High-Frequency Filtering)
* **Theory**: Sensor noise and ISO grain manifest as high-frequency pseudo-random variations across neighboring pixels.
* **Mechanism**: We apply a Gaussian smoothing filter to suppress high-frequency noise, creating a reference low-frequency image. Subtracting this smoothed image from the original grayscale image isolates high-frequency residual noise.

### 4. Corruption & Artifact Detection (Boundary Histogram Clipping)
* **Theory**: Digital transmission errors, sensor sensor burnouts, and patch corruptions often produce unnatural clusters of solid black ($0$) or solid white ($255$) pixels.
* **Mechanism**: We measure the fraction of total pixels located at the extreme boundaries of the 8-bit intensity histogram.

---

## 🧠 AI / Machine Learning Model Architecture

The machine learning component is implemented in **PyTorch** as a Multi-Task Multilayer Perceptron (`QualityAssessmentMLP`):

```
Feature Vector (Sharpness, Brightness, Contrast, Noise, Saturation, Clipping) [6]
                                 │
                                 ▼
                     Linear Layer (6 ──▶ 32) + ReLU
                                 │
                                 ▼
                     Linear Layer (32 ──▶ 16) + ReLU
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Quality Regression Head]       [Defect Classification Head]
         Linear (16 ──▶ 1) + Sigmoid     Linear (16 ──▶ 5) + Sigmoid
                 │                               │
                 ▼                               ▼
      Overall Quality Score [0-100]    Probabilities: Blur, Underexposure,
                                       Overexposure, Noise, Corruption
```

### Multi-Task Loss Objective:
The network is optimized using a combined composite loss function:
* **Quality Score Regression**: Mean Squared Error ($\text{MSE}$) between predicted and ground-truth quality scores.
* **Defect Classification**: Multi-label Binary Cross-Entropy ($\text{BCE}$) across the 5 independent defect probability logits.
* **Joint Optimization**: The defect loss is weighted ($2.0 \times$) relative to the regression loss to ensure the network strictly penalizes missed defect identification.

---

## 🔄 Dataset & Synthetic Degradation Pipeline

To ensure 100% reproducibility without requiring gigabytes of external dataset downloads, the platform features a self-contained synthetic dataset generator:

### 1. Base Image Acquisition (`download_images.py`)
* Automatically fetches 20 high-resolution clean base images via the Picsum REST API.
* **Procedural Fallback**: If external internet access is unavailable or rate-limited during deployment/testing, the script automatically generates synthetic textured images using geometric primitives and color gradients.

### 2. Synthetic Degradation Transformations (`train.py`)
From each clean base image, the pipeline programmatically generates 30 controlled variations by injecting precise physical degradations:
* **Blur**: Variable Gaussian kernels ($5\times 5$, $11\times 11$, $15\times 15$, $21\times 21$).
* **Underexposure**: Down-scaling intensity values ($30\% - 60\%$).
* **Overexposure**: Up-scaling intensities ($150\% - 200\%$) with positive additive bias.
* **Gaussian Noise**: Additive zero-mean Gaussian distribution with variable standard deviation.
* **Patch Corruption**: Random rectangular black-out masks simulating transmission packet loss.

### 3. Balanced Training
Each degradation mathematically decrements the ground-truth quality score and sets corresponding defect binary labels, yielding a balanced dataset split ($80\%$ training, $20\%$ validation) for supervised learning.

---

## 🗺 Explainability & Spatial Heatmaps (Localization)

Beyond returning numerical scores, the system localizes problematic regions:

1. **Patch-based Grid Analysis**: The image is partitioned into a $16 \times 16$ grid of $32 \times 32$ pixel patches.
2. **Localized Edge Density**: The Laplacian variance is computed independently for each patch.
3. **Inversion & Normalization**: The variance grid is normalized and inverted so that regions of highest degradation (loss of sharpness/detail) receive the maximum thermal index.
4. **Colormap Blending**: The thermal grid is mapped through OpenCV's `COLORMAP_JET` and alpha-blended ($60\%$ original, $40\%$ heatmap overlay).
5. **Base64 Transmission**: The resulting overlay is transmitted as a Base64-encoded image for real-time visualization on the frontend.

---

## 📂 Codebase Architecture & File-by-File Guide

```
iiit_assignment/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI application entrypoint & CORS middleware
│   │   ├── api.py                # REST route handlers (/analyze, /analyze-batch, /history, /health)
│   │   ├── database.py           # SQLAlchemy SQLite database session manager
│   │   ├── models.py             # SQLAlchemy ORM database models
│   │   └── ml/
│   │       ├── model.py          # PyTorch Multi-Task MLP neural network definition
│   │       ├── cv_features.py    # OpenCV feature extraction & spatial heatmap generator
│   │       ├── inference.py      # End-to-end inference wrapper & defect thresholding
│   │       ├── train.py          # Synthetic dataset generation & PyTorch training routine
│   │       └── cli_inference.py  # Standalone CLI utility for headless image evaluation
│   ├── download_images.py        # Base image downloader with procedural fallback
│   ├── requirements.txt          # Python package dependencies
│   ├── conftest.py               # Pytest path resolution configuration
│   ├── Dockerfile                # Multi-stage backend container definition
│   └── tests/
│       └── test_api.py           # Automated test suite (routes, ML inference, DB)
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main React UI component (Dropzone, Score Meter, Heatmap viewer)
│   │   ├── index.css             # Vanilla CSS design tokens, glassmorphism & dark-mode styling
│   │   └── main.jsx              # React DOM root mounting
│   ├── package.json              # Frontend dependencies (React, Vite, Axios, Lucide)
│   ├── Dockerfile                # Multi-stage Nginx production container definition
│   └── index.html                # HTML entrypoint
├── .github/
│   └── workflows/
│       └── main.yml              # GitHub Actions CI/CD automated test & build workflow
├── docker-compose.yml            # Multi-container orchestration specification
├── .gitignore                    # Root git exclusion rules
└── README.md                     # Comprehensive project documentation
```

### Detailed Component Roles:
* [`backend/app/ml/cv_features.py`](backend/app/ml/cv_features.py): Extracts numerical statistics (sharpness, brightness, contrast, noise, saturation, clipping) and generates the JET colormap heatmap.
* [`backend/app/ml/model.py`](backend/app/ml/model.py): Defines the dual-head PyTorch neural network.
* [`backend/app/ml/inference.py`](backend/app/ml/inference.py): Loads the trained `model.pth`, executes the feature extraction, runs tensor forward passes, and applies confidence thresholding.
* [`backend/app/api.py`](backend/app/api.py): Provides endpoints for single-image upload, batch processing, database logging, and historical query retrieval.
* [`frontend/src/App.jsx`](frontend/src/App.jsx): React component featuring drag-and-drop batch ingestion, dynamic radial SVG score meters, defect severity badges, and side-by-side thermal heatmap comparison.

---

## 📡 REST API Specification

### 1. Analyze Single Image
* **POST** `/api/v1/analyze`
* **Content-Type**: `multipart/form-data`
* **Request Body**: `file` (Binary Image File)

#### Response Example (`200 OK`):
```json
{
  "id": 1,
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [
    {
      "type": "noise",
      "severity": "low",
      "confidence": 0.71
    }
  ],
  "stats": {
    "sharpness": 0.854,
    "brightness": 0.602,
    "contrast": 0.451,
    "noise_level": 0.298,
    "saturation": 0.548,
    "clipping_fraction": 0.002
  },
  "heatmap": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### 2. Analyze Batch of Images
* **POST** `/api/v1/analyze-batch`
* **Content-Type**: `multipart/form-data`
* **Request Body**: `files` (Array of Binary Image Files)
* **Response**: Returns a JSON object containing `batch_results` array with individual analysis records.

### 3. Analysis History
* **GET** `/api/v1/history?skip=0&limit=50`
* **Response**: Array of historical database records ordered chronologically.

### 4. Health Check
* **GET** `/api/v1/health`
* **Response**: `{"status": "ok"}`

---

## 🚀 Quick Start & Setup Guide

### 1. Start Backend Service
```powershell
cd backend

# Create & activate Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
# On macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download base images & train model (one-time setup)
python download_images.py
python app/ml/train.py

# Launch FastAPI server
uvicorn app.main:app --reload --port 8000
```
* Interactive API Documentation (Swagger UI): **[http://localhost:8000/docs](http://localhost:8000/docs)**

### 2. Start Frontend UI (In a new terminal)
```powershell
cd frontend
npm install
npm run dev
```
* Web Application Interface: **[http://localhost:5173](http://localhost:5173)**

---

## 🖥 CLI Inference Utility

For headless environments, automated batch scripts, or quick evaluations without starting the web server:

```powershell
cd backend
python app/ml/cli_inference.py path/to/image.jpg --save-heatmap heatmap_output.jpg
```

---

## 🧪 Automated Testing & Verification

The test suite validates database isolation, API routes, batch handling, and invalid file rejection:

```powershell
cd backend
python -m pytest tests/ -v
```

### Automated Checks:
* `test_health_check`: Validates HTTP 200 health probe.
* `test_analyze_image`: Tests single-image upload, database persistence, quality score generation, and base64 heatmap validity.
* `test_analyze_batch`: Tests multi-part batch endpoints and response formatting.
* `test_analyze_invalid_file`: Tests HTTP 400 rejection for non-image payloads.
* `test_get_history`: Tests database retrieval of past records.
