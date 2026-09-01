import os
import sys
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.database import Base, get_db

# Use an isolated in-memory or separate file test database
TEST_DB_FILE = "./test_sql_app.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{TEST_DB_FILE}"
test_engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists(TEST_DB_FILE):
        try:
            os.remove(TEST_DB_FILE)
        except Exception:
            pass

@pytest.fixture(scope="module")
def sample_image():
    img_path = "test_image.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, 'Test', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(img_path, img)
    yield img_path
    if os.path.exists(img_path):
        try:
            os.remove(img_path)
        except Exception:
            pass

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_image(sample_image):
    with open(sample_image, "rb") as f:
        response = client.post("/api/v1/analyze", files={"file": ("test_image.jpg", f, "image/jpeg")})
    
    assert response.status_code == 200
    data = response.json()
    assert "quality_score" in data
    assert "quality_label" in data
    assert "issues" in data
    assert "stats" in data
    assert "heatmap" in data
    assert data["heatmap"].startswith("data:image/jpeg;base64,")

def test_analyze_batch(sample_image):
    with open(sample_image, "rb") as f1, open(sample_image, "rb") as f2:
        response = client.post("/api/v1/analyze-batch", files=[
            ("files", ("test1.jpg", f1, "image/jpeg")),
            ("files", ("test2.jpg", f2, "image/jpeg"))
        ])
        
    assert response.status_code == 200
    data = response.json()
    assert "batch_results" in data
    assert len(data["batch_results"]) == 2
    assert data["batch_results"][0]["filename"] == "test1.jpg"
    assert "quality_score" in data["batch_results"][0]

def test_analyze_invalid_file():
    response = client.post("/api/v1/analyze", files={"file": ("test.txt", b"hello world", "text/plain")})
    assert response.status_code == 400

def test_get_history():
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
