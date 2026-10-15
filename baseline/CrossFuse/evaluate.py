#!/usr/bin/env python3
"""Held-out, end-to-end CrossFuse + YOLO11n evaluation for PACE.

Run from any working directory:
  python baseline/CrossFuse/evaluate.py --split test --max-samples 100 --device auto

Uses the SAME fusion computation as train_conv_trans.py (Trans_FuseNet.train_module,
shift_flag=False), followed by YOLO. The alternative Trans_FuseNet.forward()
implements a different patch/shift path; it is deliberately not used here.

Dependencies: repository requirements plus torchmetrics[detection], pycocotools.
Outputs are saved under baseline/CrossFuse/evaluation/<run-name>/.
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
from ultralytics.utils.nms import non_max_suppression

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from network.net_autoencoder import Auto_Encoder_single
from network.net_conv_trans import Trans_FuseNet
from network.loss import Gradient_loss, Order_loss
from train_conv_trans import (custom_config, custom_config_auto, CLASS_TO_ID,
                              ID_TO_CLASS, load_targets, targets_to_metric_format,
                              calculate_metrics)
from tools import utils


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', type=Path, default=REPO / 'processed_dataset')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--max-samples', type=int, default=100, help='0 means all samples')
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    p.add_argument('--image-size', type=int, default=256, help='Square size used during training')
    p.add_argument('--rgb-weights', type=Path, default=HERE/'models/autoencoder/auto_encoder_epoch_4_rgb.model')
    p.add_argument('--ir-weights', type=Path, default=HERE/'models/autoencoder/auto_encoder_epoch_4_ir.model')
    p.add_argument('--fusion-weights', type=Path, default=HERE/'models/transfuse/fusetrans_epoch_28.model')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--confidence', type=float, default=0.5, help='Threshold for P/R/F1; AP uses 0.001')
    p.add_argument('--nms-iou', type=float, default=0.7)
    p.add_argument('--visualizations', type=int, default=12, help='Save first N fixed sample panels; 0 disables')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--no-fusion-metrics', action='store_true')
    p.add_argument('--no-latency', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def device_for(name):
    if name != 'auto':
        if name == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if name == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable')
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def load_weights(model, path, device):
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_models(args, device):
    ir = load_weights(Auto_Encoder_single(**custom_config_auto), args.ir_weights, device)
    rgb = load_weights(Auto_Encoder_single(**custom_config_auto), args.rgb_weights, device)
    fused = load_weights(Trans_FuseNet(**custom_config), args.fusion_weights, device)
    for model in (ir, rgb, fused):
        model.requires_grad_(False)
    return ir, rgb, fused


def read_metadata(path):
    if not path.is_file():
        print(f'Warning: metadata not found: {path}; subset breakdowns unavailable')
        return {}
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    return {(r.get('split', ''), Path(r.get('pair_id', '')).stem): r for r in rows}


def sample_paths(root, split, maximum):
    paths = [Path(p) for p in utils.list_dronevehicle_images(str(root), 'ir', (split,), None)]
    if maximum > 0:
        paths = paths[:maximum]
    if not paths:
        raise RuntimeError(f'No images found under {root / split / "ir"}')
    for p in paths:
        rgb = root / split / 'rgb' / p.name
        if not rgb.is_file():
            raise FileNotFoundError(f'RGB counterpart missing for {p}: {rgb}')
    return paths


def image_tensor(path, size, device):
    # Match utils.get_train_images(..., flag=False), including OpenCV grayscale conversion.
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(arr.copy()).float().unsqueeze(0).unsqueeze(0).to(device), arr


def fused_numpy(tensor):
    return tensor.detach().float().cpu()[0, 0].clamp(0, 255).numpy().astype(np.uint8)


def entropy(img):
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def spatial_frequency(img):
    a = img.astype(np.float32)
    rf = np.sqrt(np.mean(np.diff(a, axis=0)**2))
    cf = np.sqrt(np.mean(np.diff(a, axis=1)**2))
    return float(np.hypot(rf, cf))


def average_gradient(img):
    a = img.astype(np.float32)
    gx = np.diff(a, axis=1)[:-1, :]
    gy = np.diff(a, axis=0)[:, :-1]
    return float(np.sqrt((gx**2 + gy**2)/2).mean())


def mutual_information(a, b):
    # 64-bin empirical MI, in bits, with matched binning across images.
    joint, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=64, range=[[0, 256], [0, 256]])
    joint /= max(joint.sum(), 1)
    pa, pb = joint.sum(axis=1), joint.sum(axis=0)
    independent = pa[:, None] * pb[None, :]
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / independent[nz])))


def fusion_metrics(rgb, ir, fused):
    return {'entropy': entropy(fused), 'spatial_frequency': spatial_frequency(fused),
            'average_gradient': average_gradient(fused),
            'mi_rgb': mutual_information(fused, rgb), 'mi_ir': mutual_information(fused, ir),
            'mi_total': mutual_information(fused, rgb) + mutual_information(fused, ir)}


def decode(detector, image, confidence, nms_iou):
    raw = detector(image)
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    result = non_max_suppression(raw, conf_thres=confidence, iou_thres=nms_iou,
                                 nc=len(CLASS_TO_ID), max_det=300)[0]
    return {'boxes': result[:, :4].detach().cpu(),
            'scores': result[:, 4].detach().cpu(),
            'labels': result[:, 5].long().detach().cpu()}


def prediction_records(pred):
    return [{'box_xyxy': [float(v) for v in box], 'score': float(score),
             'class_id': int(label), 'class_name': ID_TO_CLASS[int(label)]}
            for box, score, label in zip(pred['boxes'].tolist(), pred['scores'].tolist(), pred['labels'].tolist())]


def draw_panel(rgb, ir, fused, predictions, targets, confidence):
    h, w = fused.shape
    left = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    middle = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR)
    for cls, cx, cy, bw, bh in targets:
        x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
        x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
        cv2.rectangle(right, (x1,y1), (x2,y2), (255, 150, 0), 1)
    for p in prediction_records(predictions):
        if p['score'] < confidence:
            continue
        x1,y1,x2,y2 = [int(x) for x in p['box_xyxy']]
        cv2.rectangle(right, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(right, f"{p['class_name']} {p['score']:.2f}", (x1, max(12,y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    header = np.zeros((28, 3*w, 3), dtype=np.uint8)
    for idx, name in enumerate(('RGB', 'IR', 'Fused + boxes (green=pred, blue=GT)')):
        cv2.putText(header, name, (idx*w+5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255,255,255), 1)
    return np.vstack((header, np.hstack((left,middle,right))))


def aggregate(values):
    vals = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if not len(vals):
        return None
    return {'mean': float(vals.mean()), 'median': float(np.median(vals)),
            'p95': float(np.percentile(vals,95)), 'min': float(vals.min()), 'max': float(vals.max())}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, value):
    with path.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def metric_engine(device):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:
        raise RuntimeError('Install torchmetrics and pycocotools: pip install torchmetrics pycocotools') from exc
    return MeanAveragePrecision(box_format='xyxy', iou_type='bbox',
                                iou_thresholds=[round(0.5+i*0.05,2) for i in range(10)],
                                class_metrics=True).to(device)


def metric_result(metric):
    r = metric.compute()
    result = {'map50': float(r['map_50'].item()), 'map50_95': float(r['map'].item())}
    for name in ('map_per_class', 'mar_100_per_class'):
        if name in r:
            result[name] = {ID_TO_CLASS.get(int(c), str(int(c))): float(v)
                            for c, v in zip(r['classes'].tolist(), r[name].tolist())}
    return result


def main():
    a = arguments()
    if a.max_samples < 0 or a.visualizations < 0 or a.warmup < 0:
        raise ValueError('Sample count, visualization count and warmup must be nonnegative')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = device_for(a.device)
    out = a.output or HERE / 'evaluation' / f'baseline_epoch_28_{a.split}_{"all" if a.max_samples == 0 else a.max_samples}'
    out.mkdir(parents=True, exist_ok=True)
    if a.visualizations:
        (out/'visualizations').mkdir(exist_ok=True)
    paths = sample_paths(a.dataset_root, a.split, a.max_samples)
    meta = read_metadata(a.dataset_root/'metadata.csv')
    print(f'Device: {device}; split: {a.split}; samples: {len(paths)}; output: {out}')
    ir_encoder, rgb_encoder, model = load_models(a, device)
    gra_loss = Gradient_loss(custom_config['out_channels']).to(device)
    order_loss = Order_loss(custom_config['out_channels']).to(device)
    metric = metric_engine(device)
    subset_metrics = {'clean': metric_engine(device), 'degraded': metric_engine(device)}
    metric_records = defaultdict(list)
    preds_for_pr, targets_for_pr = [], []
    rows, predictions_json = [], []
    latencies = defaultdict(list)
    attention_ms = []
    # Hooks time the full cross_encoder, including self- and cross-attention.
    hook_start = [None]
    def before_attention(_module, _inputs):
        sync(device)
        hook_start[0] = time.perf_counter()
    def after_attention(_module, _inputs, _output):
        sync(device)
        attention_ms.append((time.perf_counter()-hook_start[0])*1000)
    hooks = []
    if not a.no_latency:
        hooks = [model.cross_atten_block.register_forward_pre_hook(before_attention),
                 model.cross_atten_block.register_forward_hook(after_attention)]
    try:
        with torch.inference_mode():
            for i, ir_path in enumerate(paths):
                sample_id = ir_path.stem
                rgb_path = a.dataset_root/a.split/'rgb'/ir_path.name
                rgb, rgb_arr = image_tensor(rgb_path, a.image_size, device)
                ir, ir_arr = image_tensor(ir_path, a.image_size, device)
                targets = load_targets(str(ir_path), str(rgb_path), a.image_size, a.image_size)
                record = meta.get((a.split, sample_id), {})
                degraded = str(record.get('degraded', '')).strip().lower() in ('true','1','yes')
                subset = 'degraded' if degraded else 'clean' if record else 'unknown'

                # Warm up on the first sample. Not included in benchmark or predictions.
                if i == 0:
                    for _ in range(a.warmup):
                        ish, ide = ir_encoder(ir)
                        rsh, rde = rgb_encoder(rgb)
                        temp = model.train_module(ir, rgb, ish, rsh, ide, rde,
                                                  False, gra_loss, order_loss)
                        _ = decode(model.detector, temp['out']/255., 0.001, a.nms_iou)
                    sync(device)
                    attention_ms.clear()

                sync(device)
                start = time.perf_counter()
                ish, ide = ir_encoder(ir)
                rsh, rde = rgb_encoder(rgb)
                sync(device)
                encoder_end = time.perf_counter()
                output = model.train_module(ir, rgb, ish, rsh, ide, rde,
                                            False, gra_loss, order_loss)
                fused = output['out']/255.0
                sync(device)
                fusion_end = time.perf_counter()
                pred = decode(model.detector, fused, 0.001, a.nms_iou)
                sync(device)
                detection_end = time.perf_counter()
                duration = {'encoder_ms': (encoder_end-start)*1000,
                            'fusion_ms': (fusion_end-encoder_end)*1000,
                            'detection_nms_ms': (detection_end-fusion_end)*1000,
                            'end_to_end_ms': (detection_end-start)*1000}
                if not a.no_latency:
                    for k,v in duration.items(): latencies[k].append(v)
                gt = targets_to_metric_format([targets], a.image_size, a.image_size, device)[0]
                metric.update([{k:v.to(device) for k,v in pred.items()}], [gt])
                if subset in subset_metrics:
                    subset_metrics[subset].update([{k:v.to(device) for k,v in pred.items()}], [gt])
                preds_for_pr.append(pred)
                targets_for_pr.append({k:v.cpu() for k,v in gt.items()})
                fm = fusion_metrics(rgb_arr, ir_arr, fused_numpy(output['out'])) if not a.no_fusion_metrics else {}
                row = {'sample_id': sample_id, 'split': a.split, 'subset': subset,
                       'degraded': record.get('degraded', ''),
                       'modality': record.get('modality', ''),
                       'degradation_type': record.get('degradation_type', ''),
                       'severity': record.get('severity', ''),
                       'num_ground_truth': len(targets),
                       'num_predictions_at_confidence': sum(float(v)>=a.confidence for v in pred['scores']),
                       **fm, **({} if a.no_latency else duration)}
                rows.append(row)
                predictions_json.append({'sample_id':sample_id, 'predictions':prediction_records(pred),
                                         'ground_truth':[{'class_id':int(c), 'cx':float(cx), 'cy':float(cy),
                                                          'width':float(w), 'height':float(h)}
                                                         for c,cx,cy,w,h in targets]})
                if i < a.visualizations:
                    panel = draw_panel(rgb_arr, ir_arr, fused_numpy(output['out']), pred, targets, a.confidence)
                    cv2.imwrite(str(out/'visualizations'/f'{sample_id}.png'), panel)
                if (i+1)%10 == 0 or i+1 == len(paths):
                    print(f'Processed {i+1}/{len(paths)}')
    finally:
        for hook in hooks: hook.remove()

    pr = calculate_metrics(preds_for_pr, targets_for_pr, confidence=a.confidence, iou_threshold=0.5)
    overall = metric_result(metric)
    subset_report = {}
    for group, eng in subset_metrics.items():
        group_rows = [r for r in rows if r['subset']==group]
        if group_rows:
            p = [x for x,r in zip(preds_for_pr, rows) if r['subset']==group]
            t = [x for x,r in zip(targets_for_pr, rows) if r['subset']==group]
            subset_report[group] = {'count':len(group_rows), **metric_result(eng),
                                    **calculate_metrics(p,t,confidence=a.confidence,iou_threshold=0.5)}
    # Additional degradation breakdowns. Small subsets can yield unstable AP,
    # so report sample counts alongside each metric.
    breakdowns = {}
    for field in ('modality', 'degradation_type', 'severity'):
        values = sorted({str(r.get(field, '')).strip() for r in rows
                         if r['subset'] == 'degraded' and str(r.get(field, '')).strip()})
        breakdowns[field] = {}
        for value in values:
            indices = [j for j,r in enumerate(rows)
                       if r['subset'] == 'degraded' and str(r.get(field, '')).strip() == value]
            if not indices:
                continue
            sub_preds = [preds_for_pr[j] for j in indices]
            sub_targets = [targets_for_pr[j] for j in indices]
            breakdowns[field][value] = {'count':len(indices),
                **calculate_metrics(sub_preds, sub_targets, confidence=a.confidence, iou_threshold=0.5)}
    fusion_keys = ('entropy','spatial_frequency','average_gradient','mi_rgb','mi_ir','mi_total')
    fusion_summary = {k:aggregate([r.get(k) for r in rows]) for k in fusion_keys} if not a.no_fusion_metrics else {}
    latency_summary = {k:aggregate(v) for k,v in latencies.items()} if not a.no_latency else {}
    if not a.no_latency:
        latency_summary['cross_attention_block_ms'] = aggregate(attention_ms)
        latency_summary['fps_from_mean_latency'] = 1000/np.mean(latencies['end_to_end_ms'])
        if device.type=='cuda':
            latency_summary['cuda_peak_allocated_mb'] = torch.cuda.max_memory_allocated(device)/1024**2
    parameters = {'total':sum(p.numel() for p in model.parameters()),
                  'detector':sum(p.numel() for p in model.detector.parameters()),
                  'fusion_without_detector':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('detector.')),
                  'rgb_encoder':sum(p.numel() for p in rgb_encoder.parameters()),
                  'ir_encoder':sum(p.numel() for p in ir_encoder.parameters())}
    report = {'configuration':{'split':a.split,'max_samples':a.max_samples,'samples_evaluated':len(rows),
                               'device':str(device),'image_size':a.image_size,'confidence':a.confidence,
                               'nms_iou':a.nms_iou,'seed':a.seed,'warmup':a.warmup,
                               'fusion_weights':str(a.fusion_weights.resolve()),
                               'rgb_weights':str(a.rgb_weights.resolve()),
                               'ir_weights':str(a.ir_weights.resolve()),
                               'inference_path':'train_module(shift_flag=False) + detector',
                               'note':'Latency includes device synchronization and Python overhead; batch=1.'},
              'detection':{**overall,**pr}, 'subsets':subset_report, 'degradation_breakdowns':breakdowns,
              'fusion_quality':fusion_summary,'latency':latency_summary,'parameters':parameters}
    json_write(out/'summary.json',report)
    json_write(out/'latency.json',latency_summary)
    json_write(out/'predictions.json',predictions_json)
    write_csv(out/'per_sample.csv',rows)
    write_csv(out/'per_class.csv', [{'class_name':k, 'ap50_95':v,
                                     'recall_100':overall.get('mar_100_per_class',{}).get(k)}
                                    for k,v in overall.get('map_per_class',{}).items()])
    print(json.dumps({'detection':report['detection'], 'subsets':subset_report,
                      'latency':latency_summary}, indent=2))
    print(f'Wrote evaluation to: {out}')


if __name__ == '__main__':
    main()
