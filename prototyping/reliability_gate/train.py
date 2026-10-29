"""Train PACE's reliability gate with fused-patch quality pseudo-labels.

From the PACE repository root:
    python prototyping/reliability_gate/train.py --max-train 50 --max-val 20 --epochs 3
    python prototyping/reliability_gate/train.py

Stages: --stage labels (build cached labels), --stage train (train from cache),
        --stage both (default). --overwrite-labels regenerates caches.

The original TransFuse (NOT selective TransFuse) creates the fused image.
The frozen RGB quality estimator scores the grayscale fused image; this is a
provisional pseudo-labeling assumption, not a calibrated fused-image score.
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
CF = ROOT / "baseline" / "CrossFuse"
QE = ROOT / "prototyping" / "quality_estimator"
ENC = ROOT / "prototyping" / "encoder"
sys.path.insert(0, str(CF))
sys.path.insert(0, str(QE))
sys.path.insert(0, str(ENC))

from network.net_conv_trans import Trans_FuseNet
from encoder import load_encoder, extract_features
from model import SpatialQualityEstimator
from degradation_features import degradation_features
from reliability_gate import ReliabilityGate

FUSION_CONFIG = dict(
    img_size=32, patch_size=2, en_out_channels1=32,
    out_channels=1, part_out=128, train_flag=False,
    depth_self=1, depth_cross=1, n_heads=16,
    mlp_ratio=4, qkv_bias=True, p=0.0, attn_p=0.0,
    num_classes=5,
)
CLASSIFIER_PATH = CF / "models" / "transfuse" / "fusetrans_epoch_28.model"
RGB_ESTIMATOR_PATH = QE / "checkpoints" / "quality_rgb.pt"
DATA_ROOT = ROOT / "processed_dataset"
OUTPUT_ROOT = ROOT / "prototyping" / "reliability_gate" / "checkpoints"


def device_default():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_state(path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    result = torch.load(path, map_location=device, weights_only=True)
    if isinstance(result, dict) and "model_state_dict" in result:
        return result["model_state_dict"], result
    return result, {}


def load_frozen_models(device):
    ir_encoder = load_encoder("ir", device)
    rgb_encoder = load_encoder("rgb", device)

    transfuse = Trans_FuseNet(**FUSION_CONFIG)
    state, _ = checkpoint_state(CLASSIFIER_PATH, device)
    transfuse.load_state_dict(state, strict=True)
    transfuse.to(device).eval().requires_grad_(False)

    qe_state, metadata = checkpoint_state(RGB_ESTIMATOR_PATH, device)
    in_channels = int(metadata.get("in_channels", 132))
    feature_scale = float(metadata.get("extra_feature_scale", 10.0))
    quality_estimator = SpatialQualityEstimator(in_channels)
    quality_estimator.load_state_dict(qe_state, strict=True)
    quality_estimator.to(device).eval().requires_grad_(False)

    for model in (rgb_encoder, ir_encoder):
        model.eval().requires_grad_(False)
    return ir_encoder, rgb_encoder, transfuse, quality_estimator, feature_scale


def list_samples(split, limit=None):
    quality_dir = DATA_ROOT / split / "quality_maps"
    rgb_dir = DATA_ROOT / split / "rgb"
    ir_dir = DATA_ROOT / split / "ir"
    if not all(path.is_dir() for path in (quality_dir, rgb_dir, ir_dir)):
        raise FileNotFoundError(f"Missing {split} images or quality_maps under {DATA_ROOT}")

    samples = []
    for rgb_quality_path in sorted(quality_dir.glob("*_rgb.npy")):
        stem = rgb_quality_path.stem.removesuffix("_rgb")
        ir_quality_path = quality_dir / f"{stem}_ir.npy"
        rgb_path = rgb_dir / f"{stem}.jpg"
        ir_path = ir_dir / f"{stem}.jpg"
        if all(p.is_file() for p in (ir_quality_path, rgb_path, ir_path)):
            samples.append((stem, rgb_path, ir_path, rgb_quality_path, ir_quality_path))
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise RuntimeError(f"No complete RGB/IR training pairs in {split}")
    return samples


def read_grayscale(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot open {path}")
    # Fixed dimensions also align the 8x8 quality cells with 8x8 output blocks.
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image.copy()).to(device).float()[None, None] / 255.0


def read_quality(path, device):
    value = np.load(path).astype(np.float32)
    if value.shape != (8, 8) or not np.isfinite(value).all():
        raise ValueError(f"Invalid quality map {path}: {value.shape}")
    if value.min() < 0 or value.max() > 1:
        raise ValueError(f"Quality map outside [0, 1]: {path}")
    return torch.from_numpy(value)[None, None].to(device)


def generate_labels(split, limit, device, overwrite=False, margin=0.0):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = "all" if limit is None else f"first{limit}"
    cache_path = OUTPUT_ROOT / f"gate_labels_{split}_{suffix}.pt"
    if cache_path.exists() and not overwrite:
        print(f"Using existing {split} labels: {cache_path}")
        return cache_path

    samples = list_samples(split, limit)
    ir_encoder, rgb_encoder, transfuse, scorer, scale = load_frozen_models(device)
    inputs, targets = [], []
    positive_count = 0

    with torch.inference_mode():
        for index, (stem, rgb_path, ir_path, rgb_qpath, ir_qpath) in enumerate(samples, 1):
            rgb = read_grayscale(rgb_path, device)
            ir = read_grayscale(ir_path, device)
            rgb_q = read_quality(rgb_qpath, device)
            ir_q = read_quality(ir_qpath, device)

            ir_sh, ir_de = extract_features(ir_encoder, ir * 255.0)
            rgb_sh, rgb_de = extract_features(rgb_encoder, rgb * 255.0)

            # Original full TransFuse, with no gate. It internally runs YOLO;
            # we ignore detections because labels are based on image quality.
            fused = transfuse(ir_de, ir_sh, rgb_de, rgb_sh, True)["out"]
            fused_01 = (fused / 255.0).clamp(0, 1)

            # The quality estimator expects the same 132-channel preprocessing
            # used in prototyping/quality_estimator/train.py.
            _, fused_deep = extract_features(rgb_encoder, fused_01 * 255.0)
            extra = degradation_features(fused_01)
            extra = F.adaptive_avg_pool2d(extra, fused_deep.shape[-2:]) * scale
            fused_features = torch.cat((fused_deep.float(), extra.float()), dim=1)
            fused_q = scorer(fused_features)

            # "Fuse" iff estimated fused quality beats both modality labels.
            label = (fused_q > torch.maximum(rgb_q, ir_q) + margin).float()
            inputs.append(torch.cat((rgb_q, ir_q), dim=1).squeeze(0).cpu())
            targets.append(label.squeeze(0).cpu())
            positive_count += int(label.sum().item())
            if index % 25 == 0 or index == len(samples):
                print(f"{split}: labeled {index}/{len(samples)} image pairs")

    x = torch.stack(inputs)    # [N, 2, 8, 8]
    y = torch.stack(targets)   # [N, 1, 8, 8]
    torch.save({"inputs": x, "targets": y, "margin": margin}, cache_path)
    print(f"Saved {cache_path} | fuse labels: {positive_count}/{y.numel()} "
          f"({positive_count/y.numel():.1%})")
    return cache_path


def load_cache(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    x, y = data["inputs"].float(), data["targets"].float()
    if x.ndim != 4 or x.shape[1:] != (2, 8, 8) or y.shape != (len(x), 1, 8, 8):
        raise ValueError(f"Unexpected cached tensors in {path}")
    return TensorDataset(x, y)


def train_gate(train_cache, val_cache, device, epochs, batch_size, lr, seed):
    train_data = load_cache(train_cache)
    val_data = load_cache(val_cache)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    positives = float(train_data.tensors[1].sum())
    count = train_data.tensors[1].numel()
    if positives == 0 or positives == count:
        print("WARNING: Training labels contain only one class. "
              "Check fused-quality score calibration before interpreting the gate.")
    pos_weight = min((count - positives) / max(positives, 1.0), 20.0)

    gate = ReliabilityGate().to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    best = float("inf")
    model_path = OUTPUT_ROOT / "reliability_gate_best.pt"

    for epoch in range(1, epochs + 1):
        metrics = []
        for training, loader in ((True, train_loader), (False, val_loader)):
            gate.train(training)
            total_loss, total_correct, total_count = 0.0, 0, 0
            with torch.set_grad_enabled(training):
                for inputs, target in loader:
                    inputs, target = inputs.to(device), target.to(device)
                    # Use logits directly: BCEWithLogitsLoss is stable and
                    # avoids differentiating through hard threshold/STE.
                    logits = gate.network(inputs)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, target,
                        pos_weight=torch.tensor(pos_weight, device=device),
                    )
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    total_loss += float(loss.detach()) * target.numel()
                    total_correct += int(((logits.detach() > 0) == target.bool()).sum())
                    total_count += target.numel()
            metrics.append((total_loss / total_count, total_correct / total_count))

        (tr_loss, tr_acc), (val_loss, val_acc) = metrics
        print(f"Epoch {epoch:03d}/{epochs} | train BCE={tr_loss:.4f} "
              f"acc={tr_acc:.2%} | val BCE={val_loss:.4f} acc={val_acc:.2%}")
        if val_loss < best:
            best = val_loss
            torch.save({
                "model_state_dict": gate.state_dict(),
                "hidden_dim": 16,
                "epoch": epoch,
                "val_bce": val_loss,
                "val_accuracy": val_acc,
                "positive_fraction": positives / count,
                "training_method": "full_transfuse_fused_quality_pseudolabels",
                "fused_image_scorer": "quality_rgb.pt (out-of-distribution)",
                "seed": seed,
            }, model_path)

    print(f"Saved best gate: {model_path}")
    print("Use ReliabilityGate().load_state_dict(checkpoint['model_state_dict'])")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("both", "labels", "train"), default="both")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--overwrite-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_default()
    print(f"Device: {device}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix_train = "all" if args.max_train is None else f"first{args.max_train}"
    suffix_val = "all" if args.max_val is None else f"first{args.max_val}"
    train_cache = OUTPUT_ROOT / f"gate_labels_train_{suffix_train}.pt"
    val_cache = OUTPUT_ROOT / f"gate_labels_val_{suffix_val}.pt"

    if args.stage in ("labels", "both"):
        train_cache = generate_labels("train", args.max_train, device,
                                     args.overwrite_labels, args.margin)
        val_cache = generate_labels("val", args.max_val, device,
                                   args.overwrite_labels, args.margin)
    if args.stage in ("train", "both"):
        if not train_cache.exists() or not val_cache.exists():
            raise FileNotFoundError("Cached labels missing. Run --stage labels first.")
        train_gate(train_cache, val_cache, device, args.epochs,
                   args.batch_size, args.lr, args.seed)


if __name__ == "__main__":
    main()
