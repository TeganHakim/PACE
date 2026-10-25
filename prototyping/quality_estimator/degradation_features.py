
import torch
import torch.nn.functional as F


def degradation_features(image):
    """
    Compute four fixed image-statistic feature maps.

    Args:
        image: [B, 1, H, W], grayscale, values in [0, 1]

    Returns:
        [B, 4, H, W]
    """
    x = image.float()

    sobel_x = x.new_tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]).reshape(1, 1, 3, 3) / 8.0

    sobel_y = sobel_x.transpose(-1, -2)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    gradient = torch.sqrt(gx**2 + gy**2 + 1e-8)

    lap_kernel = x.new_tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]).reshape(1, 1, 3, 3)

    laplacian = F.conv2d(
        x, lap_kernel, padding=1
    ).abs()

    mean = F.avg_pool2d(
        x, kernel_size=7, stride=1, padding=3
    )
    mean_sq = F.avg_pool2d(
        x * x, kernel_size=7, stride=1, padding=3
    )
    variance = (mean_sq - mean**2).clamp_min(0)

    blurred = F.avg_pool2d(
        x, kernel_size=5, stride=1, padding=2
    )
    residual = (x - blurred).abs()

    return torch.cat(
        [gradient, laplacian, variance, residual],
        dim=1
    )
