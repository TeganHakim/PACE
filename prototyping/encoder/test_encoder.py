"""
Testing the pretrained encoder's performance on test IR and RGB images.
"""

from pathlib import Path
import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from encoder import load_encoder, extract_features

PROTOTYPING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSFUSE_ROOT = REPO_ROOT / "baseline" / "CrossFuse"
sys.path.insert(0, str(CROSSFUSE_ROOT))

from tools import utils

IR_IMAGE_PATH = PROTOTYPING_ROOT / "test_images" / "00001_ir.jpg"
RGB_IMAGE_PATH = PROTOTYPING_ROOT / "test_images" / "00001_rgb.jpg"

def load_image(path, height=256, width=256, rgb_flag=False):
    """
    Intakes an image path and resize height and width and outputs a tensor of dim [B, C, H, W]
    """

    """
    Lets look into this
    Both pretrained CrossFuse models expect images to have just 1 channel (greyscale) for both IR and RGB images.
    In the paper, they say how RGB images are converted to greyscale before being passed into the encoder as its better
    """
    image = utils.get_image(
        str(path),
        height=height,
        width=width,
        flag=rgb_flag # If True, it would attempt to use 3 channels
    )

    # H,W -> 1,H,W
    image = torch.from_numpy(image).float().unsqueeze(0)

    # 1,H,W -> 1,1,H,W
    image = image.unsqueeze(0)

    return image

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # load the pretrained CrossFuse encoders
    encoder_ir = load_encoder("ir", device)
    encoder_rgb = load_encoder("rgb", device)

    # Check first convolutional layer to see exactly the shape that it expects images to be in.
    for name, layer in encoder_ir.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            print("IR first conv:", name)
            print("Input channels:", layer.in_channels)
            print("Output channels:", layer.out_channels)
            break

    for name, layer in encoder_rgb.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            print("RGB first conv:", name)
            print("Input channels:", layer.in_channels)
            print("Output channels:", layer.out_channels)
            break

    # Create the correctly formatted tensors from the image files
    image_ir = load_image(IR_IMAGE_PATH).to(device)
    image_rgb = load_image(RGB_IMAGE_PATH).to(device)

    # Extract the CrossFuse features
    sf_ir, df_ir = extract_features(encoder_ir, image_ir)  # shallow features (sf) and deep features (df)
    sf_rgb, df_rgb = extract_features(encoder_rgb, image_rgb)

    # Verify everything worked.
    print("\nIR")
    print("Input:   ", image_ir.shape)
    print("Shallow: ", sf_ir.shape)
    print("Deep:    ", df_ir.shape)

    print("\nVisible")
    print("Input:   ", image_rgb.shape)
    print("Shallow: ", sf_rgb.shape)
    print("Deep:    ", df_rgb.shape)

if __name__ == "__main__":
    main()
