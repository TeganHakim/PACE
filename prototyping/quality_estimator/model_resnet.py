
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


class SpatialQualityEstimator(nn.Module):
    """
    Predicts an 8x8 patch-level quality map from an encoder feature map.

    Input:
        features: Tensor [B, C, H, W]

    Output:
        quality: Tensor [B, 1, 8, 8]
                 Values in [0, 1].
    """

    def __init__(self, in_channels):
        super().__init__()

        # Load ImageNet-pretrained ResNet-18.
        self.resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Adapt the first convolution to the encoder channels.
        old_conv = self.resnet.conv1

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize using pretrained weights.
        with torch.no_grad():
            pretrained_weights = old_conv.weight

            mean_weights = pretrained_weights.mean(
                dim=1,
                keepdim=True
            )

            if in_channels == 3:
                new_conv.weight.copy_(pretrained_weights)

            else:
                new_conv.weight.copy_(
                    mean_weights.repeat(
                        1, in_channels, 1, 1
                    ) * (3.0 / in_channels)
                )

        self.resnet.conv1 = new_conv

        # Remove classification components.
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        # Spatial quality decoder.
        self.decoder = nn.Sequential(
            nn.Conv2d(
                512, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 1,
                kernel_size=1
            )
        )

    def forward(self, x):

        # ResNet-18 feature extraction.
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Predict spatial quality.
        x = self.decoder(x)

        # Match ground-truth quality map dimensions.
        x = F.interpolate(
            x,
            size=(8, 8),
            mode="bilinear",
            align_corners=False
        )

        # Quality scores between 0 and 1.
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":

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
    print(
        "Output range:",
        output.min().item(),
        output.max().item()
    )

    assert output.shape == (batch_size, 1, 8, 8)

    print("Model test passed.")
