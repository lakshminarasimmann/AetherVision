import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import AnalysisResult
from app.ml.inference import run_inference

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def is_valid_image(file: UploadFile) -> bool:
    if file.content_type:
        return file.content_type.startswith("image/")
    ext = os.path.splitext(file.filename or "")[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"]

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not is_valid_image(file):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        # Save file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run ML inference
        result = run_inference(file_path)
        
        # Save to database
        db_result = AnalysisResult(
            filename=file.filename,
            quality_score=result["quality_score"],
            quality_label=result["quality_label"],
            issues=result["issues"],
            stats=result["stats"]
        )
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Ensure we return the exact format expected by the assessment
        return {
            "id": db_result.id,
            "quality_score": result["quality_score"],
            "quality_label": result["quality_label"],
            "issues": result["issues"],
            "stats": result["stats"],
            "heatmap": result.get("heatmap")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-batch")
async def analyze_batch(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    results = []
    for file in files:
        if not is_valid_image(file):
            results.append({"filename": file.filename, "error": "Not an image file"})
            continue
            
        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            result = run_inference(file_path)
            
            db_result = AnalysisResult(
                filename=file.filename,
                quality_score=result["quality_score"],
                quality_label=result["quality_label"],
                issues=result["issues"],
                stats=result["stats"]
            )
            db.add(db_result)
            db.commit()
            
            if os.path.exists(file_path):
                os.remove(file_path)
                
            results.append({
                "filename": file.filename,
                "quality_score": result["quality_score"],
                "quality_label": result["quality_label"],
                "issues": result["issues"],
                "stats": result["stats"],
                "heatmap": result.get("heatmap")
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
            
    return {"batch_results": results}

@router.get("/history")
def get_history(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    results = db.query(AnalysisResult).order_by(AnalysisResult.timestamp.desc()).offset(skip).limit(limit).all()
    return results

@router.get("/health")
def health_check():
    return {"status": "ok"}
