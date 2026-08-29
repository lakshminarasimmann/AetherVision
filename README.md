# AI-Powered Image Quality & Defect Detection

![CI/CD Status](https://github.com/your-username/repo-name/actions/workflows/main.yml/badge.svg)

This is a complete, production-ready full-stack application that accepts images and automatically evaluates their visual quality. Designed to demonstrate a meaningful and explainable use of Computer Vision and Machine Learning, the system employs a **Hybrid AI Approach** to detect common degradation issues and localize them using **Quality Heatmaps**.

## ✨ Key Features & "Big Implementations"

1. **Quality Heatmaps (Localization)**: Instead of just returning a global score, the engine computes a spatial grid of localized feature variances, applies a JET colormap, and returns a visual heatmap overlay revealing *exactly* where the image is degraded.
2. **Batch Processing Architecture**: The backend (`/analyze-batch`) and the frontend drag-and-drop zone fully support concurrent multi-image analysis, accelerating dataset triage.
3. **Hybrid AI Model**: A powerful blend of deterministic feature extraction (OpenCV) and a learned neural network (PyTorch MLP) ensures accurate and highly interpretable results without the "black box" nature of massive CNNs.
4. **CI/CD Pipeline**: Configured GitHub Actions workflow (`.github/workflows/main.yml`) that automatically runs linting, tests, and Docker builds.
5. **Automated Testing Suite**: Built-in `pytest` coverage for the backend API endpoints and image parsing logic.
6. **Premium UI**: Developed with React and Vanilla CSS, featuring glassmorphism, micro-animations, and dynamic result rendering.

## 🧠 Architectural Deep Dive

### The Hybrid Approach
Traditional pure-DL models (like fine-tuned ResNets) often suffer from lack of interpretability. To fulfill the assessment requirements while maintaining explainability, this project implements a hybrid feature-based ML approach:

1. **Computer Vision Extraction (OpenCV)**
   - **Sharpness**: Measured via the *Variance of the Laplacian* ($ \text{Var}(\nabla^2 I) $). Blurred images lack high-frequency edges, leading to a low variance.
   - **Exposure**: Computed using the Mean and Standard Deviation of the HSV Value (V) channel.
   - **Noise**: Estimated by taking the absolute difference between the original grayscale image and a Gaussian-blurred version.
   - **Corruption**: Measured by histogram clipping at absolute thresholds (0 and 255).

2. **Multilayer Perceptron (PyTorch)**
   - The exact numeric features extracted above are normalized and passed into a fully-connected PyTorch neural network.
   - The network outputs a global Quality Score (0-100) and defect probabilities (Blur, Noise, Over/Under Exposure) via sigmoid activations.

3. **Dynamic Training & Dataset Generation**
   - **Is Training Required?**: Yes, the PyTorch MLP must be trained to map OpenCV features to quality scores. However, to ensure the application is easily reproducible and runnable without downloading gigabytes of external datasets, the training is **completely automated**.
   - **The Dataset**: We use a dynamic, synthetic dataset generation approach. The `download_images.py` script fetches 20-30 high-quality, royalty-free placeholder images (from Picsum). These act as our "clean" base images (Quality Score = 100).
   - **Synthetic Degradation Pipeline**: The `train.py` script iterates over these clean images and artificially applies severe degradations using OpenCV:
     - *Blur*: Applying variable kernel-size Gaussian Blurs.
     - *Exposure*: Linearly scaling pixel intensities up (Overexposure) or down (Underexposure).
     - *Noise*: Injecting Gaussian noise into the pixel arrays.
     - *Corruption*: Masking out random blocks of the image.
   - **Why Synthetic?**: This method guarantees a perfectly balanced dataset of both acceptable and defective images. It allows the neural network to explicitly learn the correlation between the OpenCV features and the specific defect we injected. The model weights (`model.pth`) are then saved for inference.

### System Connectivity (Frontend ↔ Backend)

The application is built on a decoupled, client-server architecture:

1. **Frontend (React/Vite)**
   - The user interface is a Single Page Application (SPA).
   - When a user uploads images, React stores them in browser memory using `URL.createObjectURL` for immediate preview generation.
   - **Axios HTTP Client**: To communicate with the backend, the frontend constructs a `FormData` object. The image files are appended to this object.
   - Axios sends an asynchronous `POST` request to the backend with the `Content-Type` set to `multipart/form-data`.

2. **Backend (FastAPI)**
   - The FastAPI server listens on port `8000`. It utilizes `python-multipart` to parse the incoming binary image streams.
   - **CORS (Cross-Origin Resource Sharing)**: A `CORSMiddleware` is configured in `main.py` to allow the React frontend (running on a different port/domain) to securely make requests to the API.
   - The image is temporarily saved to disk, processed by the OpenCV + PyTorch inference pipeline, and the results (Quality Score, Defect Labels, and Heatmap Base64) are saved to the **SQLite** database via **SQLAlchemy**.
   - Finally, FastAPI serializes the database record and the heatmap string into a standard JSON payload and returns it to the React frontend, which reactively updates the UI to display the Results Dashboard.

---

## 🚀 Setup & Deployment

### Option 1: One-Click Docker Compose (Recommended)
This approach containerizes the frontend (Nginx) and backend (FastAPI + PyTorch) into an isolated environment.

1. Ensure Docker and Docker Compose are installed.
2. Run the stack:
   ```bash
   docker-compose up --build
   ```
   > **Note**: During the first build, the backend Dockerfile will run the `download_images.py` and `train.py` scripts to generate the synthetic dataset and train the model weights locally.
3. Access the Application:
   - **Frontend UI**: [http://localhost](http://localhost) (or `http://localhost:5173` if running React dev server natively)
   - **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Native Development Setup

#### Backend (Python 3.11)
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# 1. Download base training images
python download_images.py

# 2. Train Model
python app/ml/train.py

# 3. Start Server
uvicorn app.main:app --reload
```

#### Frontend (Node 20)
```bash
cd frontend
npm install
npm run dev
```

---

## 📡 REST API Reference

The backend exposes a structured JSON REST API powered by FastAPI.

### 1. Single Image Analysis
`POST /api/v1/analyze`
Accepts `multipart/form-data` with a single `file` field.

**Response Example:**
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
    "sharpness": 0.85,
    "brightness": 0.60,
    "contrast": 0.45,
    "noise_level": 0.30,
    "saturation": 0.55,
    "clipping_fraction": 0.01
  },
  "heatmap": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

### 2. Batch Analysis
`POST /api/v1/analyze-batch`
Accepts `multipart/form-data` with multiple `files` fields. Returns an array of results under the `batch_results` key.

### 3. History
`GET /api/v1/history`
Returns paginated history of past analyses stored in the SQLite database.

---

## 🧪 Testing
The backend is fully equipped with an automated Pytest suite.
```bash
cd backend
pytest tests/
```
The tests validate DB connections, file handling exceptions, and inference pipeline stability.
