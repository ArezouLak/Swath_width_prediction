import torch
import torch.nn as nn
import torchvision.models as models


class CNNFeatureExtractor(nn.Module):
    """
    Generic CNN backbone for frame-level feature extraction.

    Supported backbones:
    - resnet18
    - efficientnet_b0
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            self.feature_dim = model.fc.in_features
            self.backbone = nn.Sequential(*list(model.children())[:-1])

        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            self.feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            self.backbone = model

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        returns: [B, feature_dim]
        """
        features = self.backbone(x)

        if features.dim() == 4:
            features = features.flatten(1)

        return features
