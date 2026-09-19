"""
Testing the pretrained encoder's performance on test IR and RGB images.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from encoder import load_encoder, extract_features

PROTOTYPING_ROOT = Path(__file__).resolve().parents[1]

image_ir_PATH = PROTOTYPING_ROOT / "test_images" / "00001_ir.jpg"
RGB_IMAGE_PATH = PROTOTYPING_ROOT / "test_images" / "00001_rgb.jpg"

def load_image(path):
    """ 
    Produce a tensor of the expected representation dimensions for CrossFuse

    Intakes an image path and outputs a tensor of dim [B, C, H, W]
    """

    image = Image.open(path).convert("L")
    image = ToTensor()(image) * 255.0  # CrossFuse does /255 internally so scale the 0,1 from TF accordingly
    image = image.unsqueeze(0)

    return image


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # load the pretrained CrossFuse encoders
    encoder_ir = load_encoder("rgb", device)
    encoder_rgb = load_encoder("ir", device)

    # Create the correctly formatted tensors from the image files
    image_ir = load_image(image_ir_PATH).to(device)
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


