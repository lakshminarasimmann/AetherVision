import os
import torch
import numpy as np
from app.ml.model import QualityAssessmentMLP
from app.ml.cv_features import extract_features

# Global variable to hold the loaded model
_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defect names corresponding to indices 0-4
DEFECT_NAMES = ["blur", "underexposure", "overexposure", "noise", "corruption"]

def load_model():
    global _MODEL
    if _MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        _MODEL = QualityAssessmentMLP()
        
        if os.path.exists(model_path):
            _MODEL.load_state_dict(torch.load(model_path, map_location=_DEVICE))
        else:
            print(f"Warning: Model weights not found at {model_path}. Using random weights.")
            
        _MODEL.to(_DEVICE)
        _MODEL.eval()

def run_inference(image_path: str):
    """
    Runs the full inference pipeline (CV extraction + PyTorch inference)
    Returns:
        - overall quality score (0-100 int)
        - quality label (str)
        - list of detected issues
        - raw features dict
    """
    load_model()
    
    # 1. Extract CV features
    feats_array, feats_dict, heatmap_base64 = extract_features(image_path)
    
    # 2. PyTorch Inference
    x = torch.tensor(feats_array, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    
    with torch.no_grad():
        q_pred, d_pred = _MODEL(x)
        
    quality_score_float = q_pred.item()
    defects_probs = d_pred.squeeze(0).cpu().numpy()
    
    # Scale quality score to 0-100
    quality_score = int(quality_score_float * 100)
    
    # Determine label
    if quality_score >= 80:
        quality_label = "ACCEPTABLE"
    elif quality_score >= 50:
        quality_label = "DEGRADED"
    else:
        quality_label = "DEFECTIVE"
        
    # Detect issues
    issues = []
    for i, prob in enumerate(defects_probs):
        if prob > 0.4: # Threshold for detection
            if prob > 0.8:
                severity = "high"
            elif prob > 0.6:
                severity = "medium"
            else:
                severity = "low"
                
            issues.append({
                "type": DEFECT_NAMES[i],
                "severity": severity,
                "confidence": round(float(prob), 2)
            })
            
    # Add potential visual defect dynamically if score is very low but no specific defect matched
    if quality_score < 40 and len(issues) == 0:
        issues.append({
            "type": "potential visual defect",
            "severity": "high",
            "confidence": 0.85
        })
        
    return {
        "quality_score": quality_score,
        "quality_label": quality_label,
        "issues": issues,
        "stats": feats_dict,
        "heatmap": heatmap_base64
    }
