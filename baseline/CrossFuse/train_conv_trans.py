# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Project: FuseTrans
# @File: train.py

import os
import random
import xml.etree.ElementTree as ET

import scipy.io as scio
import torch
from torch.optim import Adam
from tqdm import tqdm

from tools import utils
from ultralytics.utils.nms import non_max_suppression
from types import SimpleNamespace
from network.net_conv_trans import Trans_FuseNet
from network.net_autoencoder import Auto_Encoder_single
from network.loss import Gradient_loss, Order_loss
from args_trans import Args as args

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    MeanAveragePrecision = None


# -------------------------------------------------------
# Model configuration
# -------------------------------------------------------

custom_config_auto = {
    "in_channels": 1,
    "out_channels": 1,
    "en_out_channels1": 32,
    "en_out_channels": 64,
    "num_layers": 3,
    "dense_out": 128,
    "part_out": 128,
    "train_flag": True,
}

custom_config = {
    "en_out_channels1": 32,
    "out_channels": 1,
    "part_out": 128,
    "train_flag": True,
    "img_size": 32,
    "patch_size": 2,
    "depth_self": 1,
    "depth_cross": 1,
    "n_heads": 16,
    "qkv_bias": True,
    "mlp_ratio": 4,
    "p": 0.0,
    "attn_p": 0.0,
    "num_classes": 5,
}

CLASS_TO_ID = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "van": 3,
    "freight car": 4,
}
NUM_CLASSES = len(CLASS_TO_ID)
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


# -------------------------------------------------------
# Annotation loading
# -------------------------------------------------------

def get_annotation_path(image_path, modality="ir"):
    marker = f"/{modality}/"
    replacement = (
        "/ir_annotations/" if modality == "ir"
        else "/rgb_annotations/"
    )
    if marker not in image_path.replace("\\", "/"):
        raise ValueError(
            f"Cannot map {modality!r} image path: {image_path}"
        )

    normalized_path = image_path.replace("\\", "/")
    annotation_path = normalized_path.replace(marker, replacement, 1)
    return os.path.splitext(annotation_path)[0] + ".xml"


def load_xml_boxes(xml_path):
    """Load polygon annotations as pixel-coordinate xyxy boxes."""
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(xml_path)

    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    boxes = []

    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in CLASS_TO_ID:
            continue

        polygon = obj.find("polygon")
        if polygon is None:
            continue

        try:
            xs = [float(polygon.findtext(f"x{i}")) for i in range(1, 5)]
            ys = [float(polygon.findtext(f"y{i}")) for i in range(1, 5)]
        except (TypeError, ValueError):
            continue

        x1, x2 = max(0.0, min(xs)), min(width, max(xs))
        y1, y2 = max(0.0, min(ys)), min(height, max(ys))

        if x2 > x1 and y2 > y1:
            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_name": name,
            })

    return boxes, width, height


def calculate_iou(box1, box2):
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1["x2"] - box1["x1"]) * max(
        0.0, box1["y2"] - box1["y1"]
    )
    area2 = max(0.0, box2["x2"] - box2["x1"]) * max(
        0.0, box2["y2"] - box2["y1"]
    )
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def merge_rgb_ir_boxes(rgb_boxes, ir_boxes, iou_threshold=0.5):
    """Merge same-class overlapping boxes and retain unmatched boxes."""
    matched_ir = set()
    merged = []

    for rgb_box in rgb_boxes:
        best_idx, best_iou = None, 0.0

        for idx, ir_box in enumerate(ir_boxes):
            if idx in matched_ir:
                continue
            if rgb_box["class_name"] != ir_box["class_name"]:
                continue

            iou = calculate_iou(rgb_box, ir_box)
            if iou > best_iou:
                best_idx, best_iou = idx, iou

        if best_idx is not None and best_iou >= iou_threshold:
            ir_box = ir_boxes[best_idx]
            matched_ir.add(best_idx)
            merged.append({
                "x1": min(rgb_box["x1"], ir_box["x1"]),
                "y1": min(rgb_box["y1"], ir_box["y1"]),
                "x2": max(rgb_box["x2"], ir_box["x2"]),
                "y2": max(rgb_box["y2"], ir_box["y2"]),
                "class_name": rgb_box["class_name"],
            })
        else:
            merged.append(dict(rgb_box))

    merged.extend(
        dict(box) for idx, box in enumerate(ir_boxes)
        if idx not in matched_ir
    )
    return merged


def load_targets(ir_path, rgb_path, image_width, image_height):
    """Load merged RGB/IR annotations in normalized cxcywh format."""
    ir_xml = get_annotation_path(ir_path, "ir")
    rgb_xml = get_annotation_path(rgb_path, "rgb")

    ir_boxes, ir_w, ir_h = load_xml_boxes(ir_xml)
    try:
        rgb_boxes, rgb_w, rgb_h = load_xml_boxes(rgb_xml)
        if abs(ir_w - rgb_w) > 1 or abs(ir_h - rgb_h) > 1:
            raise ValueError(f"RGB/IR annotation dimensions differ: {ir_path}")
        boxes = merge_rgb_ir_boxes(rgb_boxes, ir_boxes)
    except FileNotFoundError:
        # Use the available IR annotations when RGB XML is unavailable.
        boxes = ir_boxes

    targets = []
    for box in boxes:
        x1 = box["x1"] * image_width / ir_w
        x2 = box["x2"] * image_width / ir_w
        y1 = box["y1"] * image_height / ir_h
        y2 = box["y2"] * image_height / ir_h

        targets.append((
            CLASS_TO_ID[box["class_name"]],
            ((x1 + x2) / 2) / image_width,
            ((y1 + y2) / 2) / image_height,
            (x2 - x1) / image_width,
            (y2 - y1) / image_height,
        ))

    return targets


def make_yolo_batch(batch_targets, images):
    """Build the batch dictionary required by Ultralytics DetectionModel."""
    classes, boxes, batch_indices = [], [], []

    for batch_idx, targets in enumerate(batch_targets):
        for class_id, cx, cy, width, height in targets:
            classes.append([class_id])
            boxes.append([cx, cy, width, height])
            batch_indices.append(batch_idx)

    device = images.device
    return {
        "img": images,
        "cls": torch.tensor(
            classes, dtype=torch.float32, device=device
        ).reshape(-1, 1),
        "bboxes": torch.tensor(
            boxes, dtype=torch.float32, device=device
        ).reshape(-1, 4),
        "batch_idx": torch.tensor(
            batch_indices, dtype=torch.long, device=device
        ),
    }


def targets_to_metric_format(batch_targets, width, height, device):
    """Convert normalized cxcywh targets to TorchMetrics xyxy format."""
    results = []

    for targets in batch_targets:
        boxes, labels = [], []
        for class_id, cx, cy, bw, bh in targets:
            x1 = max(0.0, (cx - bw / 2) * width)
            y1 = max(0.0, (cy - bh / 2) * height)
            x2 = min(float(width), (cx + bw / 2) * width)
            y2 = min(float(height), (cy + bh / 2) * height)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        results.append({
            "boxes": torch.tensor(
                boxes, dtype=torch.float32, device=device
            ).reshape(-1, 4),
            "labels": torch.tensor(
                labels, dtype=torch.long, device=device
            ),
        })

    return results


@torch.no_grad()
def decode_predictions(detector, images):
    """Decode detector predictions for metric calculation."""
    was_training = detector.training
    detector.eval()

    try:
        raw = detector(images)

        if isinstance(raw, (tuple, list)):
            raw = raw[0]

        detections = non_max_suppression(
            raw,
            conf_thres=0.001,
            iou_thres=0.7,
            nc=NUM_CLASSES,
        )

        return [
            {
                "boxes": det[:, :4],
                "scores": det[:, 4],
                "labels": det[:, 5].long(),
            }
            for det in detections
        ]
    finally:
        detector.train(was_training)


# -------------------------------------------------------
# Dataset loading
# -------------------------------------------------------

def load_split(root, split, train_num=None):
    """Load image paths for a single split, using its actual directory."""
    paths = utils.list_dronevehicle_images(
        root, "ir", (split,), train_num
    )
    return sorted(paths)


def get_rgb_paths(ir_paths):
    return [
        path.replace("\\", "/").replace("/ir/", "/rgb/", 1)
        for path in ir_paths
    ]


def load_batch_targets(ir_paths, rgb_paths):
    targets = []
    for ir_path, rgb_path in zip(ir_paths, rgb_paths):
        targets.append(
            load_targets(ir_path, rgb_path, args.Width, args.Height)
        )
    return targets


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

def calculate_metrics(predictions, targets, confidence=0.5, iou_threshold=0.5):
    """Calculate greedy class-aware precision, recall, F1 and mean IoU."""
    tp = fp = fn = 0
    matched_ious = []

    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction["boxes"].cpu().tolist()
        pred_scores = prediction["scores"].cpu().tolist()
        pred_labels = prediction["labels"].cpu().tolist()
        true_boxes = target["boxes"].cpu().tolist()
        true_labels = target["labels"].cpu().tolist()

        matched_targets = set()
        order = sorted(
            range(len(pred_scores)),
            key=lambda idx: pred_scores[idx],
            reverse=True,
        )

        for pred_idx in order:
            if pred_scores[pred_idx] < confidence:
                continue

            box = pred_boxes[pred_idx]
            best_idx, best_iou = None, 0.0

            for idx, true_box in enumerate(true_boxes):
                if idx in matched_targets:
                    continue
                if pred_labels[pred_idx] != true_labels[idx]:
                    continue

                x1 = max(box[0], true_box[0])
                y1 = max(box[1], true_box[1])
                x2 = min(box[2], true_box[2])
                y2 = min(box[3], true_box[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area_a = max(0.0, box[2] - box[0]) * max(
                    0.0, box[3] - box[1]
                )
                area_b = max(0.0, true_box[2] - true_box[0]) * max(
                    0.0, true_box[3] - true_box[1]
                )
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0

                if iou > best_iou:
                    best_idx, best_iou = idx, iou

            if best_idx is not None and best_iou >= iou_threshold:
                tp += 1
                matched_targets.add(best_idx)
                matched_ious.append(best_iou)
            else:
                fp += 1

        fn += len(true_boxes) - len(matched_targets)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": (
            sum(matched_ious) / len(matched_ious)
            if matched_ious else 0.0
        ),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


# -------------------------------------------------------
# Training and validation
# -------------------------------------------------------

def run_epoch(
    model,
    model_auto_ir,
    model_auto_rgb,
    data,
    optimizer,
    gra_loss,
    order_loss,
    device,
    img_flag,
    epoch,
    split,
    metric_every,
):
    is_training = split == "train"
    model.train(is_training)

    if is_training:
        model_auto_ir.eval()
        model_auto_rgb.eval()

    image_paths, batch_count = utils.load_dataset(
        data, args.batch
    )

    batches_seen = 0
    total_loss_sum = 0.0
    det_loss_sum = 0.0
    fusion_loss_sum = 0.0
    metric_predictions = []
    metric_targets = []
    map_metric = None

    metric_batches = (
        set(range(batch_count)) if split == "val"
        else set(range(0, batch_count, max(1, metric_every)))
    )
    metric_batches.add(batch_count - 1)

    if MeanAveragePrecision is not None:
        map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[
                0.50, 0.55, 0.60, 0.65, 0.70,
                0.75, 0.80, 0.85, 0.90, 0.95,
            ],
            class_metrics=True,
        ).to(device)

    progress = tqdm(
        range(batch_count),
        desc=f"{split.capitalize()} {epoch + 1}/{args.epochs}",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch_idx in progress:
        start = batch_idx * args.batch
        ir_paths = image_paths[start:start + args.batch]
        if not ir_paths:
            continue

        rgb_paths = get_rgb_paths(ir_paths)

        images_ir = utils.get_train_images(
            ir_paths,
            height=args.Height,
            width=args.Width,
            flag=img_flag,
        ).to(device)

        images_rgb = utils.get_train_images(
            rgb_paths,
            height=args.Height,
            width=args.Width,
            flag=img_flag,
        ).to(device)

        targets = load_batch_targets(ir_paths, rgb_paths)

        with torch.no_grad():
            ir_sh, ir_de = model_auto_ir(images_ir)
            rgb_sh, rgb_de = model_auto_rgb(images_rgb)

        if is_training:
            outputs = model.train_module(
                images_ir,
                images_rgb,
                ir_sh,
                rgb_sh,
                ir_de,
                rgb_de,
                False,
                gra_loss,
                order_loss,
            )
        else:
            with torch.no_grad():
                outputs = model.train_module(
                    images_ir,
                    images_rgb,
                    ir_sh,
                    rgb_sh,
                    ir_de,
                    rgb_de,
                    False,
                    gra_loss,
                    order_loss,
                )

        fused = outputs["out"] / 255.0
        yolo_batch = make_yolo_batch(targets, fused)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

            det_loss, _ = model.detector(yolo_batch)
            det_loss_value = det_loss.sum()
            fusion_loss_value = outputs["total_loss"]
            loss = det_loss_value + fusion_loss_value
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                det_loss, _ = model.detector(yolo_batch)
                det_loss_value = det_loss.sum()
                fusion_loss_value = outputs["total_loss"]
                loss = det_loss_value + fusion_loss_value

        loss_value = float(loss.detach().item())
        batches_seen += 1

        progress.set_postfix(loss=f"{loss_value:.4f}")

        if batch_idx in metric_batches:
            predictions = decode_predictions(model.detector, fused.detach())
            targets_metric = targets_to_metric_format(
                targets, args.Width, args.Height, device
            )
            metric_predictions.extend(predictions)
            metric_targets.extend(targets_metric)

            if map_metric is not None:
                map_metric.update(predictions, targets_metric)
        total_loss_sum += loss.item()
        det_loss_sum += det_loss_value.item()
        fusion_loss_sum += fusion_loss_value.item()

    metrics = calculate_metrics(metric_predictions, metric_targets)
    mean_total_loss = total_loss_sum / max(batches_seen, 1)
    mean_det_loss = det_loss_sum / max(batches_seen, 1)
    mean_fusion_loss = fusion_loss_sum / max(batches_seen, 1)

    summary = (
        f"{split.capitalize()} epoch {epoch + 1}: "
        f"loss={mean_total_loss:.4f}, "
        f"det_loss={mean_det_loss:.4f}, "
        f"fusion_loss={mean_fusion_loss:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, "
        f"IoU={metrics['mean_iou']:.4f}"
    )

    if map_metric is not None:
        results = map_metric.compute()
        summary += (
            f", mAP50={results['map_50'].item():.4f}"
            f", mAP50-95={results['map'].item():.4f}"
        )

        if "map_per_class" in results and "classes" in results:
            per_class = results["map_per_class"].detach().cpu().tolist()
            class_ids = results["classes"].detach().cpu().tolist()

            summary += ", class mAP: " + ", ".join(
                f"{ID_TO_CLASS.get(int(class_id), str(class_id))}={value:.3f}"
                for class_id, value in zip(class_ids, per_class)
                if value >= 0
            )

    print(summary)
    return mean_total_loss, metrics


def train(data_root, img_flag):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Training on {device}")

    model = Trans_FuseNet(**custom_config).to(device)
    model.detector.args = SimpleNamespace(
		box=7.5,
		cls=0.5,
		dfl=1.5,
	)
    model_auto_ir = Auto_Encoder_single(**custom_config_auto).to(device)
    model_auto_rgb = Auto_Encoder_single(**custom_config_auto).to(device)

    if args.resume_model_trans:
        model.load_state_dict(
            torch.load(args.resume_model_trans, map_location=device)
        )

    if args.resume_model_auto_ir:
        model_auto_ir.load_state_dict(
            torch.load(args.resume_model_auto_ir, map_location=device)
        )
    if args.resume_model_auto_vi:
        model_auto_rgb.load_state_dict(
            torch.load(args.resume_model_auto_vi, map_location=device)
        )

    model_auto_ir.eval().requires_grad_(False)
    model_auto_rgb.eval().requires_grad_(False)

    gra_loss = Gradient_loss(custom_config["out_channels"]).to(device)
    order_loss = Order_loss(custom_config["out_channels"]).to(device)

    optimizer = Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.0005,
    )

    if MeanAveragePrecision is None:
        print("TorchMetrics unavailable; mAP will be skipped.")

    train_data = load_split(data_root, "train", args.train_num)
    val_data = load_split(data_root, "val", None)

    if not train_data:
        raise RuntimeError(f"No training images found under {data_root}/train/")
    if not val_data:
        raise RuntimeError(f"No validation images found under {data_root}/val/")

    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")

    output_root = args.save_fusion_model
    model_dir = os.path.join(output_root, "transfuse")
    loss_dir = os.path.join(model_dir, "loss")
    os.makedirs(loss_dir, exist_ok=True)

    train_history = []
    val_history = []

    # Compute detection metrics on every Nth batch.
    metric_every = 100

    print("Starting training...")
    for epoch in range(args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr)

        train_loss, _ = run_epoch(
            model,
            model_auto_ir,
            model_auto_rgb,
            train_data,
            optimizer,
            gra_loss,
            order_loss,
            device,
            img_flag,
            epoch,
            "train",
            metric_every,
        )
        val_loss, _ = run_epoch(
            model,
            model_auto_ir,
            model_auto_rgb,
            val_data,
            optimizer,
            gra_loss,
            order_loss,
            device,
            img_flag,
            epoch,
            "val",
            metric_every,
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        scio.savemat(
            os.path.join(loss_dir, "loss_data_trans.mat"),
            {
                "train_loss": train_history,
                "val_loss": val_history,
            },
        )

        checkpoint = os.path.join(
            model_dir, f"fusetrans_epoch_{epoch + 1}.model"
        )
        torch.save(model.state_dict(), checkpoint)
        print(f"Checkpoint saved: {checkpoint}")

    print("Training complete.")


if __name__ == "__main__":
    # True = RGB channels; False = grayscale.
    img_flag = args.channel != 1
    train(args.path, img_flag)