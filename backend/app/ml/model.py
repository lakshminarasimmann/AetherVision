import torch
import torch.nn as nn
import torch.nn.functional as F

class QualityAssessmentMLP(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=[32, 16]):
        super(QualityAssessmentMLP, self).__init__()
        
        # 6 features from OpenCV: sharpness, brightness, contrast, noise, saturation, clipping
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        
        # Outputs
        # 1. Overall quality score (0.0 to 1.0)
        self.quality_head = nn.Linear(hidden_sizes[1], 1)
        
        # 2. Defect Probabilities (Blur, Underexposure, Overexposure, Noise, Corruption)
        # Using 5 distinct output logits, sigmoid activation to get probabilities
        self.defect_head = nn.Linear(hidden_sizes[1], 5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        quality_score = torch.sigmoid(self.quality_head(x))
        defects = torch.sigmoid(self.defect_head(x))
        
        return quality_score, defects
