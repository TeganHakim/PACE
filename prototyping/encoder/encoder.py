"""
Wrapper for the pretrained CrossFuse encoders for both RGB and IR modalities. 
PACE reuses the pretrained baseline autoencoders for feature extraction.
"""

from pathlib import Path
import sys
import torch

# PACE/
REPO_ROOT = Path(__file__).resolve().parents[2]

# PACE/baseline/CrossFuse/
CROSSFUSE_ROOT = REPO_ROOT / "baseline" / "CrossFuse"

# CrossFuse uses imports such as `from network.loss import ...`, so its root
# directory needs to be available on Python's module search path.
sys.path.insert(0, str(CROSSFUSE_ROOT))

from network.net_autoencoder import Auto_Encoder_single

# ---------------- CrossFuse autoencoder config ---------- # 
# same architecture configuration used by CrossFuse.
ENCODER_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "en_out_channels1": 32,
    "en_out_channels": 64,
    "num_layers": 3,
    "dense_out": 128,
    "part_out": 128,
    "train_flag": True,
}


# ----------------- Loading the pretrained model ------------ # 
def load_encoder(modality, device="cpu"):
    """ 
    Load the pretrained autoencoder for the desired modality. 
    
    Args:
        modality: "ir" for infrared or "rgb" for rgb/visible
        device: pytorch device (ex: "cpu" or "cuda")
    
    Returns:
        pretrained Auto_Encoder_single in evaluation mode
    
    """

    if modality not in {"ir", "rgb"}:
        raise ValueError("modality must be either ir or rgb")

    if modality == "ir":
        weights_path = (
            CROSSFUSE_ROOT
            / "models"
            / "autoencoder"
            / f"auto_encoder_epoch_4_ir.model"
        )
    else:  # for PACE, we refer to visible information as "RGB" information, so we need to adjust accordingly
        weights_path = (
            CROSSFUSE_ROOT
            / "models"
            / "autoencoder"
            / f"auto_encoder_epoch_4_vi.model"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Could not find pretrained CrossFuse weights: {weights_path}"
        )
    
    # construct the encoder
    model = Auto_Encoder_single(**ENCODER_CONFIG)
    
    # populate it with the pretrained parameters
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    # move model to desired device 
    model.to(device)

    # encoder to eval mode as we'll use for inference (not training)
    model.eval()

    # ensure we don't change the pretrained parameters
    for parameter in model.parameters(): 
        parameter.requires_grad = False
    
    # return the pretrained encoder
    return model 


# ----------------- Extracting features with the pretrained model --------- # 
def extract_features(model, image):
    """
    Pass an image through the pretrained CrossFuse encoder

    Args:
        model: loaded CrossFuse Auto_Encoder_single
        image: Tensor with shape [B, 1, H, W] and pixel values in [0, 255]
    
    Returns:
        shallow_features: CrossFuse shallow feature map
        deep_features: CrossFuse deep feature map
    """

    with torch.no_grad():  # avoid building the computation graph, save some computation time as we're not going to be doing backprop
        # pytorch's model(x) invokes a call to forward() 
        shallow_features, deep_features = model(image) 

    return shallow_features, deep_features



