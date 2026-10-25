
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialQualityEstimator(nn.Module):
    """
    Predict an 8x8 quality map from CrossFuse encoder features.

    Input:
        features: [B, C, H, W]
                  C = 160 for shallow + deep CrossFuse features

    Output:
        quality: [B, 1, 8, 8], values in [0, 1]
    """

    def __init__(self, in_channels):
        super().__init__()

        # Load ImageNet-pretrained ResNet-18 weights.
        try:
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
        except AttributeError:
            # Compatibility with older torchvision versions.
            resnet = models.resnet18(pretrained=True)

        # Convert CrossFuse features into 3 channels for ResNet.
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                in_channels, 3,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # Keep the pretrained stem and early residual layers.
        # These preserve more spatial detail for patch prediction.
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        # ResNet layer2 outputs 128 channels.
        self.quality_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # [B, 160, H, W] -> [B, 3, H, W]
        x = self.input_projection(x)

        # For 64x64 inputs, the backbone produces 8x8 features.
        x = self.backbone(x)
        x = self.quality_head(x)

        # Ensure the required 8x8 output size.
        if x.shape[-2:] != (8, 8):
            x = F.interpolate(
                x,
                size=(8, 8),
                mode="bilinear",
                align_corners=False
            )

        return torch.sigmoid(x)


if __name__ == "__main__":
    model = SpatialQualityEstimator(in_channels=160)
    dummy_features = torch.randn(4, 160, 64, 64)

    output = model(dummy_features)

    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Input shape:", dummy_features.shape)
    print("Output shape:", output.shape)
    print("Output range:",
          output.min().item(), output.max().item())
    print(f"Trainable parameters: {parameters:,}")

    assert output.shape == (4, 1, 8, 8)
    print("Model test passed.")