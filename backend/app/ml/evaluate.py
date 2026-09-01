import os
import sys
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score
)

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.ml.model import QualityAssessmentMLP
from app.ml.train import generate_synthetic_data, SyntheticDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

DEFECT_NAMES = ["Blur", "Underexposure", "Overexposure", "Noise", "Corruption"]

def evaluate_model():
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw_images')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw_images')),
        os.path.abspath("data/raw_images")
    ]
    base_dir = next((p for p in candidates if os.path.exists(p)), candidates[0])
    
    print(f"Loading data from {base_dir}...")
    dataset = generate_synthetic_data(base_dir, num_samples_per_image=50)
    
    if not dataset:
        print("No dataset found. Generating on the fly...")
        from download_images import download_images
        download_images(output_dir=base_dir, count=20)
        dataset = generate_synthetic_data(base_dir, num_samples_per_image=30)

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    test_loader = DataLoader(SyntheticDataset(test_data), batch_size=16, shuffle=False)
    
    model = QualityAssessmentMLP()
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    
    if not os.path.exists(model_path):
        print("Model weights not found. Training model first...")
        from app.ml.train import train_model
        train_model()

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    all_q_true = []
    all_q_pred = []
    all_d_true = []
    all_d_pred = []

    with torch.no_grad():
        for feats, q_target, d_target in test_loader:
            q_pred, d_pred = model(feats)
            all_q_true.extend(q_target.squeeze().numpy())
            all_q_pred.extend(q_pred.squeeze().numpy())
            all_d_true.extend(d_target.numpy())
            all_d_pred.extend(d_pred.numpy())

    all_q_true = np.array(all_q_true)
    all_q_pred = np.array(all_q_pred)
    all_d_true = np.array(all_d_true)
    all_d_pred = np.array(all_d_pred)
    all_d_pred_binary = (all_d_pred > 0.5).astype(int)

    # 1. Quality Regression Metrics
    mse = mean_squared_error(all_q_true, all_q_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_q_true, all_q_pred)
    r2 = r2_score(all_q_true, all_q_pred)
    pearson_corr = np.corrcoef(all_q_true, all_q_pred)[0, 1]

    # 2. Multi-label Defect Classification Metrics
    accuracy = accuracy_score(all_d_true, all_d_pred_binary)
    precision_macro = precision_score(all_d_true, all_d_pred_binary, average='macro', zero_division=0)
    recall_macro = recall_score(all_d_true, all_d_pred_binary, average='macro', zero_division=0)
    f1_macro = f1_score(all_d_true, all_d_pred_binary, average='macro', zero_division=0)
    
    precision_micro = precision_score(all_d_true, all_d_pred_binary, average='micro', zero_division=0)
    recall_micro = recall_score(all_d_true, all_d_pred_binary, average='micro', zero_division=0)
    f1_micro = f1_score(all_d_true, all_d_pred_binary, average='micro', zero_division=0)
    
    roc_auc = roc_auc_score(all_d_true, all_d_pred, average='macro')

    print("\n=======================================================")
    print("       AETHERVISION AI - MODEL EVALUATION REPORT       ")
    print("=======================================================\n")
    
    print("--- 1. Quality Score Regression Metrics ---")
    print(f"Mean Squared Error (MSE):         {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):    {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):         {mae:.4f}")
    print(f"R-squared Score (R2):              {r2:.4f}")
    print(f"Pearson Correlation (PCC):         {pearson_corr:.4f}")
    
    print("\n--- 2. Multi-Label Defect Classification Metrics ---")
    print(f"Subset Accuracy (Exact Match):    {accuracy*100:.2f}%")
    print(f"Macro Precision:                  {precision_macro*100:.2f}%")
    print(f"Macro Recall:                     {recall_macro*100:.2f}%")
    print(f"Macro F1-Score:                   {f1_macro*100:.2f}%")
    print(f"Micro F1-Score:                   {f1_micro*100:.2f}%")
    print(f"Macro ROC-AUC:                    {roc_auc:.4f}")

    print("\n--- 3. Per-Defect Performance Breakdown ---")
    for i, defect in enumerate(DEFECT_NAMES):
        p = precision_score(all_d_true[:, i], all_d_pred_binary[:, i], zero_division=0)
        r = recall_score(all_d_true[:, i], all_d_pred_binary[:, i], zero_division=0)
        f = f1_score(all_d_true[:, i], all_d_pred_binary[:, i], zero_division=0)
        auc = roc_auc_score(all_d_true[:, i], all_d_pred[:, i]) if len(np.unique(all_d_true[:, i])) > 1 else 1.0
        print(f"• {defect:<16} | Precision: {p*100:5.2f}% | Recall: {r*100:5.2f}% | F1: {f*100:5.2f}% | AUC: {auc:.4f}")

if __name__ == "__main__":
    evaluate_model()
