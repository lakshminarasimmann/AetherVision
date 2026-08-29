import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random

# Fix imports since we will run this as a script
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from app.ml.cv_features import extract_features
from app.ml.model import QualityAssessmentMLP

# Defect indices: 0: Blur, 1: Underexposure, 2: Overexposure, 3: Noise, 4: Corruption
class SyntheticDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        features, quality, defects = self.data[idx]
        return torch.tensor(features, dtype=torch.float32), \
               torch.tensor([quality], dtype=torch.float32), \
               torch.tensor(defects, dtype=torch.float32)

def generate_synthetic_data(base_image_dir, num_samples_per_image=10):
    dataset = []
    
    if not os.path.exists(base_image_dir):
        print(f"Directory {base_image_dir} does not exist.")
        return dataset
        
    image_files = [f for f in os.listdir(base_image_dir) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print("No images found for synthetic data generation.")
        return dataset
        
    for img_file in image_files:
        img_path = os.path.join(base_image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Base clean image (quality: 1.0, defects: 0)
        clean_path = os.path.join(base_image_dir, "temp_clean.jpg")
        cv2.imwrite(clean_path, img)
        feats, _, _ = extract_features(clean_path)
        dataset.append((feats, 1.0, [0, 0, 0, 0, 0]))
        
        for _ in range(num_samples_per_image):
            degraded = img.copy()
            defects = [0, 0, 0, 0, 0]
            quality = 1.0
            
            # Apply random degradations
            # 1. Blur
            if random.random() < 0.3:
                k = random.choice([5, 11, 15, 21])
                degraded = cv2.GaussianBlur(degraded, (k, k), 0)
                defects[0] = 1
                quality -= 0.3
                
            # 2 & 3. Exposure
            if random.random() < 0.4:
                if random.random() < 0.5:
                    # Underexposure
                    degraded = cv2.convertScaleAbs(degraded, alpha=random.uniform(0.3, 0.6), beta=0)
                    defects[1] = 1
                    quality -= 0.3
                else:
                    # Overexposure
                    degraded = cv2.convertScaleAbs(degraded, alpha=random.uniform(1.5, 2.0), beta=50)
                    defects[2] = 1
                    quality -= 0.3
                    
            # 4. Noise
            if random.random() < 0.3:
                noise = np.random.normal(0, random.uniform(10, 50), degraded.shape).astype(np.float32)
                degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                defects[3] = 1
                quality -= 0.2
                
            # 5. Corruption
            if random.random() < 0.1:
                # Add random black boxes
                h, w = degraded.shape[:2]
                x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
                x2, y2 = x1 + random.randint(50, w//2), y1 + random.randint(50, h//2)
                degraded[y1:y2, x1:x2] = 0
                defects[4] = 1
                quality -= 0.4
                
            quality = max(0.0, quality) # Clamp quality to 0-1
            
            # Save temporarily to extract features (simulate real pipeline)
            temp_path = os.path.join(base_image_dir, "temp_degraded.jpg")
            cv2.imwrite(temp_path, degraded)
            
            try:
                feats, _, _ = extract_features(temp_path)
                dataset.append((feats, quality, defects))
            except Exception as e:
                pass
                
        # Cleanup temp
        if os.path.exists(os.path.join(base_image_dir, "temp_clean.jpg")):
            os.remove(os.path.join(base_image_dir, "temp_clean.jpg"))
        if os.path.exists(os.path.join(base_image_dir, "temp_degraded.jpg")):
            os.remove(os.path.join(base_image_dir, "temp_degraded.jpg"))
            
    return dataset

def train_model():
    print("Generating synthetic dataset...")
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw_images')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw_images')),
        os.path.abspath("data/raw_images")
    ]
    base_dir = next((p for p in candidates if os.path.exists(p)), candidates[0])
    dataset = generate_synthetic_data(base_dir, num_samples_per_image=30)
    
    if not dataset:
        print("Dataset is empty. Run download_images.py first.")
        return
        
    print(f"Generated {len(dataset)} samples. Training model...")
    
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(SyntheticDataset(train_data), batch_size=16, shuffle=True)
    val_loader = DataLoader(SyntheticDataset(val_data), batch_size=16, shuffle=False)
    
    model = QualityAssessmentMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Loss functions
    # MSE for quality score (regression)
    criterion_quality = nn.MSELoss()
    # BCE for defect probabilities (multi-label classification)
    criterion_defects = nn.BCELoss()
    
    epochs = 150
    best_val_loss = float('inf')
    
    model_save_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for feats, q_target, d_target in train_loader:
            optimizer.zero_grad()
            q_pred, d_pred = model(feats)
            
            loss_q = criterion_quality(q_pred, q_target)
            loss_d = criterion_defects(d_pred, d_target)
            loss = loss_q + 2.0 * loss_d # Weight defect loss higher
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, q_target, d_target in val_loader:
                q_pred, d_pred = model(feats)
                val_loss += criterion_quality(q_pred, q_target).item() + 2.0 * criterion_defects(d_pred, d_target).item()
                
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            
    print(f"Training complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
