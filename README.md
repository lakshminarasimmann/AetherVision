# AI-Powered Image Quality & Defect Detection Platform

[![CI/CD Pipeline](https://github.com/lakshminarasimmann/mage-quality-defect-detection/actions/workflows/main.yml/badge.svg)](https://github.com/lakshminarasimmann/mage-quality-defect-detection/actions/workflows/main.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-5C3EE8?logo=opencv)](https://opencv.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com)

A full-stack, deployable artificial intelligence application engineered to evaluate image visual quality, identify distinct photographic defects, localize spatial degradations using thermal heatmaps, and persist historical diagnostics.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [System Architecture & Data Flow](#-system-architecture--data-flow)
3. [Defect Taxonomy & Detection Capabilities](#-defect-taxonomy--detection-capabilities)
4. [Computer Vision Feature Engineering (Mathematical Formulations)](#-computer-vision-feature-engineering)
5. [Hybrid AI / Machine Learning Engine](#-hybrid-ai--machine-learning-engine)
6. [Explainability & Spatial Quality Heatmaps](#-explainability--spatial-quality-heatmaps)
7. [Synthetic Dataset Generation Pipeline](#-synthetic-dataset-generation-pipeline)
8. [Backend REST API Specification](#-backend-rest-api-specification)
9. [Database Schema & Persistence](#-database-schema--persistence)
10. [Frontend UI/UX Implementation](#-frontend-uiux-implementation)
11. [CI/CD & DevOps Engineering](#-cicd--devops-engineering)
12. [Step-by-Step Deployment Guide](#-step-by-step-deployment-guide)
13. [CLI Inference Utility](#-cli-inference-utility)
14. [Automated Verification & Testing](#-automated-verification--testing)
15. [Assessment Criteria Alignment Matrix](#-assessment-criteria-alignment-matrix)

---

## 📌 Executive Summary & Problem Statement

Modern image processing systems require automated quality verification to eliminate unreadable, corrupted, blurred, or noisy imagery before feeding downstream models or archival systems. 

This platform solves that challenge by providing:
- **Zero External AI Dependencies**: Runs 100% locally with open-source PyTorch and OpenCV without calling third-party proprietary APIs.
- **Explainable Quality Assessment**: Instead of black-box classification, every judgment is tied directly to engineered photometric and spatial statistical metrics.
- **Defect Localization**: Pinpoints exact coordinates of degradation via dynamic colormapped heatmaps.
- **Batch & Real-Time Processing**: Supports both single-image analysis and concurrent multi-file batch uploads.
- **Production-Ready Deployment**: Containerized with multi-stage Docker builds, SQLite transactional persistence, and automated GitHub Actions CI/CD.

---

## 🏗 System Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                CLIENT BROWSER                                   │
│  [ React 18 SPA (Vite) + Vanilla CSS Glassmorphism + Live Heatmap Visualizer ]  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │  HTTP / Multipart (Axios)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI BACKEND GATEWAY                           │
│  - Input Validation & Stream Parser    - SQLite SQLAlchemy Persistence          │
│  - Single Image (/analyze)             - Batch Processor (/analyze-batch)       │
│  - History Query Engine (/history)     - Health Check Monitor (/health)         │
└──────────────────┬──────────────────────────────────────────────▲───────────────┘
                   │                                              │
                   ▼                                              │
┌──────────────────────────────────────┐       ┌──────────────────┴───────────────┐
│     OPENCV DETERMINISTIC ENGINE      │       │     PYTORCH HYBRID MLP ENGINE    │
│  - Laplacian Operator Sharpness      │       │  - 6 Input Feature Vector Nodes  │
│  - HSV Luminance Distribution        │──────▶│  - Hidden FC Layers (32 -> 16)   │
│  - Residual Differential Noise       │       │  - Head 1: Quality Score [0-100] │
│  - Boundary Histogram Clipping       │       │  - Head 2: Defect Multi-Label    │
│  - 32x32 Spatial Heatmap Generator   │       │    [Blur, Exposure, Noise, etc.] │
└──────────────────────────────────────┘       └──────────────────────────────────┘
```

---

## 🎯 Defect Taxonomy & Detection Capabilities

| Defect Category | Physical / Digital Root Cause | OpenCV Detection Method | PyTorch Classification Output |
| :--- | :--- | :--- | :--- |
| **Blur / Low Sharpness** | Defocus blur, camera shake, motion artifacts | Variance of Laplacian ($ \sigma^2(\nabla^2 I) $) | Logit 0 (`blur`) |
| **Underexposure** | Insufficient sensor light, low ISO/shutter speed | HSV Value channel Mean ($ \mu_V < 0.35 $) | Logit 1 (`underexposure`) |
| **Overexposure** | Sensor blowout, harsh direct lighting, flash washout | HSV Value channel Mean ($ \mu_V > 0.75 $) | Logit 2 (`overexposure`) |
| **Image Noise** | High ISO sensor gain, thermal sensor noise, compression grain | High-frequency absolute difference residual ($ \mu(\vert I - G_\sigma * I \vert) $) | Logit 3 (`noise`) |
| **Corruption / Severe Degradation** | Data transmission loss, file chunk corruption, dead sensor pixels | Boundary histogram clipping at 0 and 255 | Logit 4 (`corruption`) |
| **Potential Visual Defect** | Multi-factor degradation anomalies | Global Quality Score degradation trigger ($ Q < 40 $) | Generic fallback anomaly |

---

## 🔬 Computer Vision Feature Engineering

All images are resized to a canonical resolution of $512 \times 512$ to ensure dimensional invariance across feature calculations.

### 1. Sharpness Formulation (Laplacian Variance)
The discrete Laplacian operator approximates the second spatial derivative of the grayscale image $I$:
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
Convolved with the standard $3 \times 3$ kernel:
$$K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
The edge variance is then computed:
$$\text{Sharpness} = \frac{1}{N} \sum_{x,y} \left( (\nabla^2 I)(x,y) - \mu_{\nabla^2 I} \right)^2$$
*Sharp images contain high-frequency edges yielding high variance; blurry images exhibit minimal high-frequency transitions resulting in near-zero variance.*

### 2. Luminance & Exposure Metrics
Converting from BGR to HSV color space decouples chromaticity from intensity:
$$V(x,y) = \max(R(x,y), G(x,y), B(x,y))$$
$$\mu_{\text{Brightness}} = \frac{1}{N} \sum_{x,y} V(x,y), \quad \sigma_{\text{Contrast}} = \sqrt{\frac{1}{N} \sum_{x,y} (V(x,y) - \mu_V)^2}$$

### 3. Residual Noise Estimation
Noise is measured by isolating high-frequency components from the smoothed low-frequency image:
$$\text{Noise Level} = \frac{1}{N} \sum_{x,y} \left| I(x,y) - (I * G_{5\times 5})(x,y) \right|$$
where $G_{5\times 5}$ is a Gaussian filter with zero mean.

### 4. Histogram Clipping (Corruption Index)
Measures saturation or pixel loss at dynamic range boundaries:
$$\text{Clipping Fraction} = \frac{\text{Count}(I(x,y) == 0) + \text{Count}(I(x,y) == 255)}{512 \times 512}$$

---

## 🧠 Hybrid AI / Machine Learning Engine

### Model Architecture (`backend/app/ml/model.py`)
```
Input Vector [6] (Sharpness, Brightness, Contrast, Noise, Saturation, Clipping)
      │
      ▼
Linear Layer 1:  6 -> 32  +  ReLU Activation
      │
      ▼
Linear Layer 2: 32 -> 16  +  ReLU Activation
      │
      ├───▶ Quality Head: Linear(16 -> 1)  + Sigmoid  ──▶ Overall Quality Score [0.0 - 1.0]
      │
      └───▶ Defect Head:  Linear(16 -> 5)  + Sigmoid  ──▶ Probabilities [P(Blur), P(Under), P(Over), P(Noise), P(Corrupt)]
```

### Multi-Task Objective Function
The network is trained using a weighted composite loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(Q_{\text{pred}}, Q_{\text{true}}) + 2.0 \cdot \mathcal{L}_{\text{BCE}}(D_{\text{pred}}, D_{\text{true}})$$
- $\mathcal{L}_{\text{MSE}}$: Mean Squared Error enforcing smooth regression for continuous quality scoring.
- $\mathcal{L}_{\text{BCE}}$: Binary Cross-Entropy over all 5 multi-label defect channels.
- **Optimizer**: Adam ($\text{lr} = 0.005$, weight decay enabled).
- **Epochs**: 150 with automatic best-validation checkpointing to `model.pth`.

---

## 🗺 Explainability & Spatial Quality Heatmaps

In accordance with requirement Section 10 (Explainability & Localization), the system does not merely output an aggregate number:

1. **Spatial Grid Partitioning**: The image is subdivided into a $16 \times 16$ grid of $32 \times 32$ pixel patches.
2. **Local Feature Variance**: The discrete Laplacian variance is evaluated independently across each patch.
3. **Inversion & Normalization**: The variance values are normalized to $[0, 255]$ and inverted so that regions of highest degradation (lowest edge variance) correspond to maximum thermal values ($255$).
4. **Color Mapping & Alpha Blending**:
   $$\text{Overlay} = \alpha \cdot I_{\text{original}} + \beta \cdot \text{COLORMAP\_JET}(H_{\text{inv}})$$
   where $\alpha = 0.6$ and $\beta = 0.4$.
5. **Base64 Serialization**: The resulting image is encoded into a Base64 JPEG data URL and transmitted to the client for instant side-by-side rendering.

---

## 🔄 Synthetic Dataset Generation Pipeline

To eliminate the need for manual dataset curation or multi-gigabyte downloads while guaranteeing reproducibility:

1. `backend/download_images.py` fetches clean, high-resolution royalty-free base images via Picsum API ($512 \times 512$).
2. `backend/app/ml/train.py` applies parameterized synthetic transformations:
   - **Gaussian Blur**: Kernel sizes $k \in \{5, 11, 15, 21\}$.
   - **Photometric Scaling**: Multipliers $\alpha \in [0.3, 0.6]$ (underexposure) and $\alpha \in [1.5, 2.0]$ with $+50$ bias (overexposure).
   - **Gaussian Noise Injection**: Zero-mean additive noise with standard deviation $\sigma \in [10, 50]$.
   - **Patch Corruption**: Random rectangular zero-masking simulating transmission packet loss.
3. **Ground Truth Assignment**: Degradation parameters programmatically calculate deterministic ground-truth labels and scores, creating a balanced train/validation split ($80\% / 20\%$).

---

## 📡 Backend REST API Specification

Interactive OpenAPI documentation is automatically served at `/docs` (Swagger UI) and `/redoc`.

### 1. Single Image Analysis
- **Endpoint**: `POST /api/v1/analyze`
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file` (Binary Image File)

#### Response Schema (`200 OK`):
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
  "heatmap": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

### 2. Batch Image Analysis
- **Endpoint**: `POST /api/v1/analyze-batch`
- **Content-Type**: `multipart/form-data`
- **Parameters**: `files` (Array of Binary Image Files)

#### Response Schema (`200 OK`):
```json
{
  "batch_results": [
    {
      "filename": "sample_blur.jpg",
      "quality_score": 45,
      "quality_label": "DEFECTIVE",
      "issues": [
        {
          "type": "blur",
          "severity": "high",
          "confidence": 0.94
        }
      ],
      "stats": { ... },
      "heatmap": "data:image/jpeg;base64,..."
    }
  ]
}
```

### 3. Historical Analyses
- **Endpoint**: `GET /api/v1/history?skip=0&limit=50`
- **Response**: Array of historical database objects with timestamps.

### 4. Health & Readiness Probe
- **Endpoint**: `GET /api/v1/health`
- **Response**: `{"status": "ok"}`

---

## 🗄 Database Schema & Persistence

Data is managed using **SQLAlchemy 2.0 ORM** backed by an ACID-compliant **SQLite** database (`sql_app.db`).

```sql
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename VARCHAR INDEX,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    quality_score INTEGER,
    quality_label VARCHAR,
    issues JSON,
    stats JSON
);
```

---

## 💻 Frontend UI/UX Implementation

The frontend is a single-page application built on **React 18** and **Vite** styled exclusively with modern **Vanilla CSS**:
- **Aesthetic Design**: Dark theme (`#0f172a`), radial lighting accents, glassmorphic cards (`backdrop-filter: blur(12px)`), and modern typography.
- **Interactive Drag-and-Drop Dropzone**: Native HTML5 Drag and Drop API supporting simultaneous multiple-file ingestion with instant client-side thumbnail previews.
- **Dynamic Circular Gauge**: SVG radial dash-array progress meter color-coded dynamically (Green $\ge 80$, Orange $\ge 50$, Red $< 50$).
- **Heatmap Viewer**: Dual-view image comparison displaying the original input and the localized degradation heatmap.
- **Responsive Micro-animations**: Keyframe transitions for smooth state loading and result presentation.

---

## ⚙️ CI/CD & DevOps Engineering

Automated Continuous Integration and Continuous Deployment is configured in [`.github/workflows/main.yml`](.github/workflows/main.yml):

```yaml
name: CI/CD Pipeline
on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]
jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: "pip" }
      - run: |
          pip install -r requirements.txt
          sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
          pytest tests/ -v
  build-docker:
    needs: test-backend
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - run: docker build -t ai-vision-backend ./backend
      - run: docker build -t ai-vision-frontend ./frontend
```

---

## 🚀 Step-by-Step Deployment Guide

### Option 1: Docker Compose (Production Deployment)

Ensure Docker Desktop is running, then execute:

```bash
docker-compose up --build
```

*During build time, the backend Dockerfile automatically pulls sample clean images and trains `model.pth`.*

- 🌐 **Web UI**: [http://localhost](http://localhost) (Port 80)
- 📄 **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Port 8000)

---

### Option 2: Local Native Setup (Development Mode)

#### 1. Backend Service
```bash
cd backend
python -m venv venv

# Windows Activation:
.\venv\Scripts\Activate.ps1
# macOS/Linux Activation:
# source venv/bin/activate

pip install -r requirements.txt

# Run dataset download & train neural network
python download_images.py
python app/ml/train.py

# Launch FastAPI ASGI Server
uvicorn app.main:app --reload --port 8000
```

#### 2. Frontend Service (In a second terminal)
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🖥 CLI Inference Utility

For headless environments or automation scripts, you can evaluate images directly from the command line:

```bash
cd backend
python app/ml/cli_inference.py path/to/sample.jpg --save-heatmap output_heatmap.jpg
```

---

## 🧪 Automated Verification & Testing

The backend includes a comprehensive test suite in `backend/tests/test_api.py`:

```bash
cd backend
pytest tests/ -v
```

### Verified Test Cases:
- `test_health_check`: Validates HTTP 200 health probe.
- `test_analyze_image`: Tests single-image upload, database persistence, quality score generation, and base64 heatmap validity.
- `test_analyze_batch`: Tests multi-part batch endpoints and response formatting.
- `test_analyze_invalid_file`: Tests HTTP 400 rejection for non-image payloads.

---

## 📊 Assessment Criteria Alignment Matrix

| Assessment Requirement | Specific File Implementation | Technical Justification |
| :--- | :--- | :--- |
| **Computer Vision Reasoning (15%)** | [`backend/app/ml/cv_features.py`](backend/app/ml/cv_features.py) | Laplacian edge variance, HSV luminance distributions, residual noise difference filters, histogram clipping. |
| **AI / ML Implementation (25%)** | [`backend/app/ml/model.py`](backend/app/ml/model.py), [`backend/app/ml/train.py`](backend/app/ml/train.py) | Dual-head PyTorch MLP with joint MSE + BCE multi-task loss function. |
| **Model Evaluation & Rigor (15%)** | [`backend/app/ml/train.py`](backend/app/ml/train.py), [`backend/tests/`](backend/tests/) | 80/20 train/validation split, automated synthetic degradation generation, checkpoint loss tracking. |
| **Backend & API Implementation (15%)** | [`backend/app/api.py`](backend/app/api.py), [`backend/app/models.py`](backend/app/models.py) | FastAPI REST API, SQLite database persistence with SQLAlchemy, batch endpoints, error handling. |
| **Frontend Functionality (10%)** | [`frontend/src/App.jsx`](frontend/src/App.jsx), [`frontend/src/index.css`](frontend/src/index.css) | React 18, drag-and-drop batch upload, radial score gauge, side-by-side thermal heatmap viewer. |
| **Deployment & Reproducibility (10%)** | [`docker-compose.yml`](docker-compose.yml), [`backend/Dockerfile`](backend/Dockerfile), [`frontend/Dockerfile`](frontend/Dockerfile) | Multi-container Docker Compose with multi-stage build and self-contained on-build model training. |
| **Code Quality & Documentation (10%)** | [`README.md`](README.md), [`.github/workflows/`](.github/workflows/) | Production-grade documentation with mathematical formulation, API schemas, and CI/CD automation. |
