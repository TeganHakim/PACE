# PACE Evaluation Metrics

Comparison of **PACE**, **CrossFuse**, and **50/50 Addition** across three evaluation categories.

## 1. System Performance

| Metric | Purpose |
|---|---|
| **Inference Latency (ms)** | Measures the time required to generate a fused image. |
| **Peak GPU Memory (MB)** | Measures GPU memory consumption during fusion. |

## 2. Detection Performance

| Metric | Purpose |
|---|---|
| **mAP@50** | Measures overall vehicle detection performance, incorporating precision and recall across confidence thresholds. |
| **Recall** | Measures the proportion of ground-truth vehicles successfully detected, particularly under sensor degradation. |
| **Detection Confidence** | Measures the confidence assigned to correctly matched vehicle detections and how it changes under degradation. |

## 3. Fusion Quality & Robustness

| Metric | Purpose |
|---|---|
| **Noise / Degradation Transfer Ratio** | Quantifies how much corruption from a degraded sensor propagates into the fused output. Lower is better. |
| **Edge F1** | Measures preservation of meaningful edges while penalizing artificial edges introduced by noise. Higher is better. |
| **Qᴀʙ/ꜰ (Edge Preservation)** | Established image-fusion metric measuring how effectively source-image edge information is retained in the fused output. Higher is better. |

## Evaluation Protocol

All three fusion methods will be evaluated under identical conditions using:

- **Clean test dataset:** Establishes baseline fusion and detection performance.
- **Realistically degraded test dataset:** Evaluates robustness when one modality is corrupted by sensor noise or localized blur.

Degraded results will be analyzed separately by degradation type (noise vs. blur) and affected modality (RGB vs. IR).

For fusion-quality metrics, clean-reference outputs will be used to measure degradation impact. Qᴀʙ/ꜰ will use clean source images to avoid rewarding artificial noise edges.

System performance will be measured using identical hardware, input resolutions, and inference settings.

## Summary

| Category | Selected Metrics |
|---|---|
| **System Performance** | Latency, Peak GPU Memory |
| **Detection Performance** | mAP@50, Recall, Detection Confidence |
| **Fusion Quality & Robustness** | Noise/Degradation Transfer Ratio, Edge F1, Qᴀʙ/ꜰ |

**Total: 8 metrics across 3 evaluation categories.**