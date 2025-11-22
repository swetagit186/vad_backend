# architectures/resnet18_slice.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18SliceVad(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
