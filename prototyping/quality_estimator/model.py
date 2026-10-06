"""
define the PyTorch spatial CNN quality estimator. 
Input = encoder feature map; output = predicted spatial quality map. 
No dataset loading or training logic here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialQualityEstimator(nn.Module):
    """
    Predicts an 8x8 patch-level quality map from an encoder feature map.

    Input:
        features: Tensor of shape [B, C, H, W]

    Output:
        quality: Tensor of shape [B, 1, 8, 8]
                 Values are in [0, 1].
    """

    def __init__(self, in_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):
        # Learn spatial degradation patterns
        x = self.features(x)

        # Match the 8x8 ground-truth quality grid
        x = F.adaptive_avg_pool2d(x, (8, 8))

        # Continuous quality score in [0, 1]
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    # Temporary test. Replace 128 with the actual encoder
    # channel dimension once the encoder is trained.
    batch_size = 4
    channels = 128
    height = 64
    width = 80

    dummy_features = torch.randn(
        batch_size,
        channels,
        height,
        width
    )

    model = SpatialQualityEstimator(
        in_channels=channels
    )

    output = model(dummy_features)

    print("Input shape: ", dummy_features.shape)
    print("Output shape:", output.shape)
    print("Output range:",
          output.min().item(),
          output.max().item())

    assert output.shape == (batch_size, 1, 8, 8)

    print("Model test passed.")