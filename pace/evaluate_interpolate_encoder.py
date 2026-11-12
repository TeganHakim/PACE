#!/usr/bin/env python3
"""Held-out, end-to-end PACE selective-fusion and YOLO11n evaluation.

Run from the PACE repository root:
    python pace/evaluate.py --split test --max-samples 100 --device auto

Pipeline: RGB/IR -> frozen encoders -> quality estimates -> reliability gate
          -> LocalCrossFuse -> one YOLO11n detection -> NMS/metrics.

Quality prediction uses original-resolution images; fusion uses 256x256.
The fused tensor is already normalized to [0,1] before YOLO inference.
Outputs retain the existing JSON/CSV names and visualization structure.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PACE_ROOT = HERE
CROSSFUSE_ROOT = REPO / 'baseline' / 'CrossFuse'
SELECTIVE_ROOT = REPO / 'prototyping' / 'selective_fusion'
sys.path.insert(0, str(CROSSFUSE_ROOT))
sys.path.insert(0, str(SELECTIVE_ROOT))
sys.path.insert(0, str(PACE_ROOT))
from inference import load_models, load_image, load_quality_image, predict_quality, samples, choose_device
from local_crossfuse import LocalCrossFuse
from train_conv_trans import CLASS_TO_ID, ID_TO_CLASS, load_targets, targets_to_metric_format, calculate_metrics


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-root', type=Path, default=REPO / 'processed_dataset')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--max-samples', type=int, default=100, help='0 means all available samples')
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--context-features', type=int, default=8)
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for precision/recall/F1')
    parser.add_argument('--nms-iou', type=float, default=0.7)
    parser.add_argument('--visualizations', type=int, default=12, help='Save first N visualization panels; 0 disables')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--no-fusion-metrics', action='store_true')
    parser.add_argument('--no-latency', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--detector-weights', type=Path, default=CROSSFUSE_ROOT / 'models' / 'transfuse' / 'fusetrans_epoch_28.model')
    return parser.parse_args()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_detector(path, device):
    """

    Extract the detector trained inside Trans_FuseNet.

    CrossFuse saved the complete Trans_FuseNet state_dict, so detector

    parameters are stored under keys beginning with 'detector.'.

    """
    if not path.is_file():
        raise FileNotFoundError('Detector checkpoint not found: {}'.format(path))
    checkpoint = safe_torch_load(path, device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if not isinstance(checkpoint, dict):
        raise TypeError('Expected a state_dict checkpoint, got {}'.format(type(checkpoint)))
    detector_state = {key[len('detector.'):]: value for key, value in checkpoint.items() if key.startswith('detector.')}
    if not detector_state:
        raise RuntimeError('No detector.* parameters were found in {}. This does not appear to be the expected full CrossFuse Trans_FuseNet checkpoint.'.format(path))
    detector = DetectionModel('yolo11n.yaml', ch=1, nc=len(CLASS_TO_ID), verbose=False).to(device)
    detector.load_state_dict(detector_state, strict=True)
    detector.eval()
    detector.requires_grad_(False)
    return detector


def read_metadata(path):
    if not path.is_file():
        print('Warning: metadata not found at {}. Clean/degraded breakdowns will be unavailable.'.format(path))
        return {}
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    return {(row.get('split', ''), Path(row.get('pair_id', '')).stem): row for row in rows}


def load_samples(split, maximum):
    sample_list = samples(split, maximum if maximum > 0 else None)
    if not sample_list:
        raise RuntimeError("No paired RGB/IR samples found for split '{}'.".format(split))
    return sample_list


@torch.inference_mode()

def prepare_sample(encoders, estimators, scales, gate_model, rgb, ir, rgb_original, ir_original):
    """

    Match the working validation script's prepare() function.

    Fusion inputs are resized to 256x256. Quality is predicted from

    original-resolution inputs. The four encoder feature tensors are

    passed into LocalCrossFuse to avoid re-encoding the images.

    """
    rgb_shallow_original, rgb_deep_original = encoders["rgb"](rgb_original * 255.0)
    ir_shallow_original, ir_deep_original = encoders["ir"](ir_original * 255.0)

    # Use original sized features for quality estimation and gate
    rgb_quality = predict_quality(estimators["rgb"], scales["rgb"], rgb_original, rgb_deep_original)
    ir_quality = predict_quality(estimators["ir"], scales["ir"], ir_original, ir_deep_original)
    gate = gate_model(rgb_quality, ir_quality)

    # Use resized (interpolated) features for fusion
    rgb_shallow = F.interpolate(rgb_shallow_original, size=(256, 256), mode="bilinear", align_corners=False)
    rgb_deep = F.interpolate(rgb_deep_original, size=(32, 32), mode="bilinear", align_corners=False)
    ir_shallow = F.interpolate(ir_shallow_original, size=(256, 256), mode="bilinear", align_corners=False)
    ir_deep = F.interpolate(ir_deep_original, size=(32, 32), mode="bilinear", align_corners=False)

    encoded_features = (rgb_shallow, rgb_deep, ir_shallow, ir_deep)
    return {'rgb_quality': rgb_quality, 'ir_quality': ir_quality, 'gate': gate, 'encoded_features': encoded_features}


@torch.inference_mode()

def run_pace(local_model, rgb, ir, prepared):
    """

    Run the same LocalCrossFuse call used by the working validation

    script. Its 'out' value is the normalized fused image [0, 1].

    """
    output = local_model(rgb, ir, prepared['rgb_quality'], prepared['ir_quality'], prepared['gate'], encoded_features=prepared['encoded_features'])
    if not isinstance(output, dict) or 'out' not in output:
        keys = list(output.keys()) if isinstance(output, dict) else type(output)
        raise RuntimeError("Expected LocalCrossFuse to return a dictionary containing 'out', matching the validation script. Received: {}".format(keys))
    fused = output['out']
    if not torch.is_tensor(fused):
        raise TypeError("LocalCrossFuse output['out'] must be a tensor; got {}".format(type(fused)))
    if fused.ndim != 4 or fused.shape[1] != 1:
        raise ValueError('Expected fused output shape [B,1,H,W], got {}'.format(tuple(fused.shape)))
    detector_input = fused.float()
    return (fused, detector_input, output)


def decode(detector, image, confidence, nms_iou):
    raw = detector(image)
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    result = non_max_suppression(raw, conf_thres=confidence, iou_thres=nms_iou, nc=len(CLASS_TO_ID), max_det=300)[0]
    return {'boxes': result[:, :4].detach().cpu(), 'scores': result[:, 4].detach().cpu(), 'labels': result[:, 5].long().detach().cpu()}


def image_numpy_01(tensor):
    """Convert a normalized [1,1,H,W] tensor to an 8-bit image."""
    return tensor.detach().float().cpu()[0, 0].clamp(0.0, 1.0).mul(255.0).round().numpy().astype(np.uint8)


def image_numpy_255(tensor):
    """Convert a [1,1,H,W] tensor in [0,255] to an 8-bit image."""
    return tensor.detach().float().cpu()[0, 0].clamp(0.0, 255.0).round().numpy().astype(np.uint8)


def fused_numpy(tensor):
    """

    Convert LocalCrossFuse output['out'] from [0,1] to uint8 [0,255].

    This conversion is only for visualization and image metrics.

    Never use this uint8 image as the detector input.

    """
    return image_numpy_01(tensor)


def entropy(image):
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def spatial_frequency(image):
    array = image.astype(np.float32)
    row_frequency = np.sqrt(np.mean(np.diff(array, axis=0) ** 2))
    column_frequency = np.sqrt(np.mean(np.diff(array, axis=1) ** 2))
    return float(np.hypot(row_frequency, column_frequency))


def average_gradient(image):
    array = image.astype(np.float32)
    gx = np.diff(array, axis=1)[:-1, :]
    gy = np.diff(array, axis=0)[:, :-1]
    return float(np.sqrt((gx ** 2 + gy ** 2) / 2).mean())


def mutual_information(first, second):
    joint, _, _ = np.histogram2d(first.ravel(), second.ravel(), bins=64, range=[[0, 256], [0, 256]])
    joint /= max(joint.sum(), 1)
    p_first = joint.sum(axis=1)
    p_second = joint.sum(axis=0)
    independent = p_first[:, None] * p_second[None, :]
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log2(joint[nonzero] / independent[nonzero])))


def fusion_metrics(rgb, ir, fused):
    mi_rgb = mutual_information(fused, rgb)
    mi_ir = mutual_information(fused, ir)
    return {'entropy': entropy(fused), 'spatial_frequency': spatial_frequency(fused), 'average_gradient': average_gradient(fused), 'mi_rgb': mi_rgb, 'mi_ir': mi_ir, 'mi_total': mi_rgb + mi_ir}


def prediction_records(prediction):
    records = []
    for box, score, label in zip(prediction['boxes'].tolist(), prediction['scores'].tolist(), prediction['labels'].tolist()):
        class_id = int(label)
        records.append({'box_xyxy': [float(value) for value in box], 'score': float(score), 'class_id': class_id, 'class_name': ID_TO_CLASS[class_id]})
    return records


def draw_panel(rgb, ir, fused, predictions, targets, confidence, gate=None):
    height, width = fused.shape
    left = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    middle = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR)
    for class_id, cx, cy, box_width, box_height in targets:
        x1 = int((cx - box_width / 2) * width)
        y1 = int((cy - box_height / 2) * height)
        x2 = int((cx + box_width / 2) * width)
        y2 = int((cy + box_height / 2) * height)
        cv2.rectangle(right, (x1, y1), (x2, y2), (255, 150, 0), 1)
    for prediction in prediction_records(predictions):
        if prediction['score'] < confidence:
            continue
        x1, y1, x2, y2 = [int(value) for value in prediction['box_xyxy']]
        cv2.rectangle(right, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(right, '{} {:.2f}'.format(prediction['class_name'], prediction['score']), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    header = np.zeros((28, 3 * width, 3), dtype=np.uint8)
    titles = ('RGB', 'IR', 'PACE fused + boxes (green=pred, blue=GT)')
    for index, title in enumerate(titles):
        cv2.putText(header, title, (index * width + 5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return np.vstack((header, np.hstack((left, middle, right))))


def aggregate(values):
    valid = np.asarray([value for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
    if not len(valid):
        return None
    return {'mean': float(valid.mean()), 'median': float(np.median(valid)), 'p95': float(np.percentile(valid, 95)), 'min': float(valid.min()), 'max': float(valid.max())}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys((key for row in rows for key in row)))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, value):
    with path.open('w') as file:
        json.dump(value, file, indent=2, allow_nan=False)


def metric_engine(device):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:
        raise RuntimeError('Install torchmetrics and pycocotools:\npip install torchmetrics pycocotools') from exc
    return MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[round(0.5 + index * 0.05, 2) for index in range(10)], class_metrics=True).to(device)


def metric_result(metric):
    result = metric.compute()
    output = {'map50': float(result['map_50'].item()), 'map50_95': float(result['map'].item())}
    for name in ('map_per_class', 'mar_100_per_class'):
        if name in result:
            output[name] = {ID_TO_CLASS.get(int(class_id), str(int(class_id))): float(value) for class_id, value in zip(result['classes'].tolist(), result[name].tolist())}
    return output


def load_evaluation_models(args, device):
    """Load the frozen PACE pipeline and pretrained CrossFuse detector once."""
    (
        encoders,
        selective,
        estimators,
        scales,
        gate_model,
    ) = load_models(
        device,
        quality_source="predicted",
        gate_mode="trained",
        context_features=args.context_features,
        batch_size=args.batch_size,
    )
    baseline = selective.baseline
    local_model = LocalCrossFuse(
        rgb_encoder=encoders["rgb"],
        ir_encoder=encoders["ir"],
        baseline=baseline,
        context_features=args.context_features,
        batch_size=args.batch_size,
    ).to(device).eval()
    detector = load_detector(args.detector_weights, device)
    detector.eval().requires_grad_(False)
    return encoders, estimators, scales, gate_model, local_model, detector


def evaluate_samples(args, device, sample_list, metadata, visualization_dir, encoders, estimators, scales, gate_model, local_model, detector):
    """Run exactly one detection per completed fused image; retain all existing evaluation outputs."""
    metric = metric_engine(device)
    subset_metrics = {
        "clean": metric_engine(device),
        "degraded": metric_engine(device),
    }
    predictions_for_pr = []
    targets_for_pr = []
    rows = []
    predictions_json = []
    latencies = defaultdict(list)
    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------
    with torch.inference_mode():
        for index, sample in enumerate(sample_list):
            (
                sample_id,
                rgb_path,
                ir_path,
                _rgb_quality_path,
                _ir_quality_path,
            ) = sample
            # Fusion-resolution inputs (normally 256x256).
            rgb = load_image(rgb_path, device)
            ir = load_image(ir_path, device)
            # Original-resolution inputs for quality prediction.
            rgb_original = load_quality_image(
                rgb_path,
                "rgb",
                device,
            )
            ir_original = load_quality_image(
                ir_path,
                "ir",
                device,
            )
            targets = load_targets(
                str(ir_path),
                str(rgb_path),
                args.image_size,
                args.image_size,
            )
            record = metadata.get(
                (args.split, Path(sample_id).stem),
                {},
            )
            degraded = (
                str(record.get("degraded", ""))
                .strip()
                .lower()
                in ("true", "1", "yes")
            )
            subset = (
                "degraded"
                if degraded
                else "clean" if record else "unknown"
            )
            # Warmup is excluded from metrics and latency.
            if index == 0:
                for _ in range(args.warmup):
                    warm_prepared = prepare_sample(
                        encoders,
                        estimators,
                        scales,
                        gate_model,
                        rgb,
                        ir,
                        rgb_original,
                        ir_original,
                    )
                    warm_fused, warm_detector_input, _ = run_pace(
                        local_model,
                        rgb,
                        ir,
                        warm_prepared,
                    )
                    _ = decode(
                        detector,
                        warm_detector_input,
                        0.001,
                        args.nms_iou,
                    )
                sync(device)
            # ---------------------------------------------
            # Preparation: encoders + quality + gate
            # ---------------------------------------------
            sync(device)
            start = time.perf_counter()
            prepared = prepare_sample(
                encoders,
                estimators,
                scales,
                gate_model,
                rgb,
                ir,
                rgb_original,
                ir_original,
            )
            sync(device)
            preparation_end = time.perf_counter()
            # ---------------------------------------------
            # PACE selective fusion
            # ---------------------------------------------
            fused, detector_input, fusion_output = run_pace(
                local_model,
                rgb,
                ir,
                prepared,
            )
            sync(device)
            fusion_end = time.perf_counter()
            # ---------------------------------------------
            # YOLO11n detection + NMS
            # ---------------------------------------------
            pred = decode(
                detector,
                detector_input,
                0.001,
                args.nms_iou,
            )
            sync(device)
            detection_end = time.perf_counter()
            duration = {
                "encoder_quality_gate_ms": (
                    preparation_end - start
                ) * 1000,
                "selective_fusion_ms": (
                    fusion_end - preparation_end
                ) * 1000,
                "detection_nms_ms": (
                    detection_end - fusion_end
                ) * 1000,
                "end_to_end_ms": (
                    detection_end - start
                ) * 1000,
            }
            if not args.no_latency:
                for key, value in duration.items():
                    latencies[key].append(value)
            # ---------------------------------------------
            # Detection metrics
            # ---------------------------------------------
            gt = targets_to_metric_format(
                [targets],
                args.image_size,
                args.image_size,
                device,
            )[0]
            metric.update(
                [
                    {
                        key: value.to(device)
                        for key, value in pred.items()
                    }
                ],
                [gt],
            )
            if subset in subset_metrics:
                subset_metrics[subset].update(
                    [
                        {
                            key: value.to(device)
                            for key, value in pred.items()
                        }
                    ],
                    [gt],
                )
            predictions_for_pr.append(pred)
            targets_for_pr.append({
                key: value.cpu()
                for key, value in gt.items()
            })
            # ---------------------------------------------
            # Fusion metrics and image conversion
            # ---------------------------------------------
            # The fused tensor is [0,1]; RGB/IR loader tensors are
            # [0,1]. Convert all three to uint8 [0,255] for metrics.
            fused_image = fused_numpy(fused)
            rgb_array = image_numpy_01(rgb)
            ir_array = image_numpy_01(ir)
            if args.no_fusion_metrics:
                fusion_report = {}
            else:
                fusion_report = fusion_metrics(
                    rgb_array,
                    ir_array,
                    fused_image,
                )
            # ---------------------------------------------
            # Gate statistics
            # ---------------------------------------------
            gate = prepared["gate"]
            active_patches = int(
                (gate >= 0.5).sum().item()
            )
            total_patches = int(gate.numel())
            active_fraction = (
                active_patches / total_patches
                if total_patches
                else 0.0
            )
            # ---------------------------------------------
            # Per-sample report
            # ---------------------------------------------
            row = {
                "sample_id": sample_id,
                "split": args.split,
                "subset": subset,
                "degraded": record.get("degraded", ""),
                "modality": record.get("modality", ""),
                "degradation_type": record.get(
                    "degradation_type",
                    "",
                ),
                "severity": record.get("severity", ""),
                "num_ground_truth": len(targets),
                "num_predictions_at_confidence": sum(
                    float(score) >= args.confidence
                    for score in pred["scores"]
                ),
                "max_prediction_score": (
                    float(pred["scores"].max())
                    if len(pred["scores"])
                    else 0.0
                ),
                "active_patches": active_patches,
                "total_patches": total_patches,
                "active_fraction": active_fraction,
                "fused_min": float(fused.min().item()),
                "fused_mean": float(fused.mean().item()),
                "fused_max": float(fused.max().item()),
                **fusion_report,
                **({} if args.no_latency else duration),
            }
            rows.append(row)
            predictions_json.append({
                "sample_id": sample_id,
                "predictions": prediction_records(pred),
                "ground_truth": [
                    {
                        "class_id": int(class_id),
                        "cx": float(cx),
                        "cy": float(cy),
                        "width": float(width),
                        "height": float(height),
                    }
                    for (
                        class_id,
                        cx,
                        cy,
                        width,
                        height,
                    ) in targets
                ],
            })
            # ---------------------------------------------
            # Visualization
            # ---------------------------------------------
            if index < args.visualizations:
                panel = draw_panel(
                    rgb_array,
                    ir_array,
                    fused_image,
                    pred,
                    targets,
                    args.confidence,
                    gate=gate,
                )
                cv2.imwrite(
                    str(
                        visualization_dir
                        / "{}.png".format(sample_id)
                    ),
                    panel,
                )
                # Save the fused image on its own as well. This
                # makes it easy to distinguish a fusion problem
                # from a panel/overlay problem.
                cv2.imwrite(
                    str(
                        visualization_dir
                        / "{}_fused.png".format(sample_id)
                    ),
                    fused_image,
                )
            if index == 0 or (index + 1) % 10 == 0:
                print(
                    "Sample {} | fused min/mean/max: "
                    "{:.4f}/{:.4f}/{:.4f} | detections: {} "
                    "(max score {:.4f})".format(
                        sample_id,
                        fused.min().item(),
                        fused.mean().item(),
                        fused.max().item(),
                        len(pred["scores"]),
                        (
                            float(pred["scores"].max())
                            if len(pred["scores"])
                            else 0.0
                        ),
                    )
                )
            if (
                (index + 1) % 10 == 0
                or index + 1 == len(sample_list)
            ):
                print(
                    "Processed {}/{}".format(
                        index + 1,
                        len(sample_list),
                    )
                )
    return metric, subset_metrics, predictions_for_pr, targets_for_pr, rows, predictions_json, latencies


def summarize_results(args, device, metric, subset_metrics, predictions_for_pr, targets_for_pr, rows, latencies):
    """Compute overall/subset detection metrics and degradation, fusion, and timing summaries."""
    pr_metrics = calculate_metrics(
        predictions_for_pr,
        targets_for_pr,
        confidence=args.confidence,
        iou_threshold=0.5,
    )
    overall_metrics = metric_result(metric)
    subset_report = {}
    for group, subset_metric in subset_metrics.items():
        group_rows = [
            row
            for row in rows
            if row["subset"] == group
        ]
        if not group_rows:
            continue
        group_predictions = [
            prediction
            for prediction, row in zip(
                predictions_for_pr,
                rows,
            )
            if row["subset"] == group
        ]
        group_targets = [
            target
            for target, row in zip(
                targets_for_pr,
                rows,
            )
            if row["subset"] == group
        ]
        subset_report[group] = {
            "count": len(group_rows),
            **metric_result(subset_metric),
            **calculate_metrics(
                group_predictions,
                group_targets,
                confidence=args.confidence,
                iou_threshold=0.5,
            ),
        }
    # -----------------------------------------------------
    # Degradation breakdowns
    # -----------------------------------------------------
    breakdowns = {}
    for field in (
        "modality",
        "degradation_type",
        "severity",
    ):
        values = sorted({
            str(row.get(field, "")).strip()
            for row in rows
            if (
                row["subset"] == "degraded"
                and str(row.get(field, "")).strip()
            )
        })
        breakdowns[field] = {}
        for value in values:
            indices = [
                index
                for index, row in enumerate(rows)
                if (
                    row["subset"] == "degraded"
                    and str(row.get(field, "")).strip() == value
                )
            ]
            sub_predictions = [
                predictions_for_pr[index]
                for index in indices
            ]
            sub_targets = [
                targets_for_pr[index]
                for index in indices
            ]
            breakdowns[field][value] = {
                "count": len(indices),
                **calculate_metrics(
                    sub_predictions,
                    sub_targets,
                    confidence=args.confidence,
                    iou_threshold=0.5,
                ),
            }
    # -----------------------------------------------------
    # Fusion quality summary
    # -----------------------------------------------------
    fusion_keys = (
        "entropy",
        "spatial_frequency",
        "average_gradient",
        "mi_rgb",
        "mi_ir",
        "mi_total",
    )
    fusion_summary = (
        {
            key: aggregate([
                row.get(key)
                for row in rows
            ])
            for key in fusion_keys
        }
        if not args.no_fusion_metrics
        else {}
    )
    # -----------------------------------------------------
    # Latency summary
    # -----------------------------------------------------
    latency_summary = (
        {
            key: aggregate(values)
            for key, values in latencies.items()
        }
        if not args.no_latency
        else {}
    )
    if not args.no_latency and latencies["end_to_end_ms"]:
        mean_latency = np.mean(latencies["end_to_end_ms"])
        latency_summary["fps_from_mean_latency"] = (
            1000.0 / mean_latency
            if mean_latency > 0
            else None
        )
        if device.type == "cuda":
            latency_summary["cuda_peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device)
                / 1024 ** 2
            )
    return pr_metrics, overall_metrics, subset_report, breakdowns, fusion_summary, latency_summary


def model_parameter_counts(encoders, estimators, gate_model, local_model, detector):
    """Retain the original parameter counting convention for report compatibility."""
    def parameter_count(module):
        return sum(
            parameter.numel()
            for parameter in module.parameters()
        )
    parameters = {
        "rgb_encoder": parameter_count(encoders["rgb"]),
        "ir_encoder": parameter_count(encoders["ir"]),
        "rgb_quality_estimator": parameter_count(
            estimators["rgb"]
        ),
        "ir_quality_estimator": parameter_count(
            estimators["ir"]
        ),
        "reliability_gate": parameter_count(gate_model),
        "local_fusion": parameter_count(local_model),
        "detector": parameter_count(detector),
    }
    parameters["total_including_detector"] = sum(
        parameters.values()
    )
    return parameters


def build_report(args, device, rows, pr_metrics, overall_metrics, subset_report, breakdowns, fusion_summary, latency_summary, parameters):
    """Build the existing report schema without renaming or removing fields."""
    report = {
        "configuration": {
            "pipeline": "PACE LocalCrossFuse + CrossFuse YOLO11n",
            "split": args.split,
            "max_samples": args.max_samples,
            "samples_evaluated": len(rows),
            "device": str(device),
            "image_size": args.image_size,
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "seed": args.seed,
            "warmup": args.warmup,
            "batch_size": args.batch_size,
            "context_features": args.context_features,
            "detector_weights": str(
                args.detector_weights.resolve()
            ),
            "inference_path": (
                "frozen encoders + predicted quality maps + "
                "trained reliability gate + LocalCrossFuse "
                "output['out'] in [0,1] + CrossFuse YOLO11n "
                "receiving the same normalized tensor"
            ),
            "note": (
                "Latency includes device synchronization and "
                "Python overhead; evaluation batch size is 1. "
                "Fused output is not divided by 255 before detection."
            ),
        },
        "detection": {
            **overall_metrics,
            **pr_metrics,
        },
        "subsets": subset_report,
        "degradation_breakdowns": breakdowns,
        "fusion_quality": fusion_summary,
        "latency": latency_summary,
        "parameters": parameters,
    }
    return report


def write_results(output_dir, report, latency_summary, predictions_json, rows, overall_metrics):
    """Write the existing JSON and CSV filenames and schemas."""
    json_write(output_dir / "summary.json", report)
    json_write(output_dir / "latency.json", latency_summary)
    json_write(output_dir / "predictions.json", predictions_json)
    write_csv(output_dir / "per_sample.csv", rows)
    write_csv(
        output_dir / "per_class.csv",
        [
            {
                "class_name": class_name,
                "ap50_95": ap,
                "recall_100": overall_metrics.get(
                    "mar_100_per_class",
                    {},
                ).get(class_name),
            }
            for class_name, ap in overall_metrics.get(
                "map_per_class",
                {},
            ).items()
        ],
    )


def main():
    """Evaluate end-to-end PACE on the requested dataset split."""
    args = arguments()
    if args.max_samples < 0 or args.visualizations < 0 or args.warmup < 0:
        raise ValueError('Sample count, visualization count, and warmup must be nonnegative.')
    if args.image_size != 256:
        print('Warning: your existing quality/fusion pipeline was configured around 256x256 fusion images.')
    set_seed(args.seed)
    device = choose_device(args.device)
    run_name = f"pace_{args.split}_{('all' if args.max_samples == 0 else args.max_samples)}"
    output_dir = args.output if args.output is not None else HERE / 'evaluation' / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / 'visualizations'
    if args.visualizations:
        visualization_dir.mkdir(parents=True, exist_ok=True)
    sample_list = load_samples(args.split, args.max_samples)
    metadata = read_metadata(args.dataset_root / 'metadata.csv')
    print(f'Device: {device}; split: {args.split}; samples: {len(sample_list)}; output: {output_dir}')
    encoders, estimators, scales, gate_model, local_model, detector = load_evaluation_models(args, device)
    metric, subset_metrics, predictions_for_pr, targets_for_pr, rows, predictions_json, latencies = evaluate_samples(args, device, sample_list, metadata, visualization_dir, encoders, estimators, scales, gate_model, local_model, detector)
    pr_metrics, overall_metrics, subset_report, breakdowns, fusion_summary, latency_summary = summarize_results(args, device, metric, subset_metrics, predictions_for_pr, targets_for_pr, rows, latencies)
    parameters = model_parameter_counts(encoders, estimators, gate_model, local_model, detector)
    report = build_report(args, device, rows, pr_metrics, overall_metrics, subset_report, breakdowns, fusion_summary, latency_summary, parameters)
    write_results(output_dir, report, latency_summary, predictions_json, rows, overall_metrics)
    print('\nEvaluation results:')
    print(json.dumps({'detection': report['detection'], 'subsets': subset_report, 'latency': latency_summary, 'fusion_quality': fusion_summary}, indent=2))
    print(f'\nWrote PACE evaluation to: {output_dir}')
if __name__ == '__main__':
    main()
