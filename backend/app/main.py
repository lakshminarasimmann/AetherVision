from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app.database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Image Quality Assessment API",
    description="API for detecting image quality and defects using CV and PyTorch.",
    version="1.0.0"
)

# Allow CORS for frontend (all origins allowed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI-Powered Image Quality & Defect Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "analyze": "POST /api/v1/analyze",
            "analyze_batch": "POST /api/v1/analyze-batch",
            "history": "GET /api/v1/history"
        }
    }

app.include_router(router, prefix="/api/v1")
