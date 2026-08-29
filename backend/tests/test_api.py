import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import os

from app.main import app
from app.database import Base, engine, get_db
from sqlalchemy.orm import sessionmaker

# Use a test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sql_app.db"
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="module")
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_sql_app.db"):
        os.remove("./test_sql_app.db")

@pytest.fixture(scope="module")
def sample_image():
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, 'Test', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite("test_image.jpg", img)
    yield "test_image.jpg"
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_image(setup_database, sample_image):
    with open(sample_image, "rb") as f:
        response = client.post("/api/v1/analyze", files={"file": ("test_image.jpg", f, "image/jpeg")})
    
    assert response.status_code == 200
    data = response.json()
    assert "quality_score" in data
    assert "heatmap" in data
    assert data["heatmap"].startswith("data:image/jpeg;base64,")

def test_analyze_batch(setup_database, sample_image):
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

def test_analyze_invalid_file(setup_database):
    response = client.post("/api/v1/analyze", files={"file": ("test.txt", b"hello world", "text/plain")})
    assert response.status_code == 400
