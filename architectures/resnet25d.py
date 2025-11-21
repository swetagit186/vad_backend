# architectures/resnet25d.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18_25D(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        old_conv = base.conv1
        base.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)

        self.model = base

    def forward(self, x):
        return self.model(x)
