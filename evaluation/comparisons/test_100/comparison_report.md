# CrossFuse vs. PACE Evaluation Report

## Evaluation runs

- **CrossFuse:** `/home/nvidia/PACE/baseline/CrossFuse/evaluation/baseline_epoch_28_test_100`
- **PACE:** `/home/nvidia/PACE/pace/evaluation/pace_test_100`
- **Output:** `/home/nvidia/PACE/evaluation/comparisons/test_100`

## Overall detection metrics

| Metric | CrossFuse | PACE | PACE − CrossFuse |
|---|---:|---:|---:|
| precision | 0.9655 | 0.9550 | -0.0105 |
| recall | 0.6750 | 0.7089 | 0.0338 |
| f1 | 0.7945 | 0.8137 | 0.0192 |
| map50 | 0.3092 | 0.3015 | -0.0077 |
| map50_95 | 0.2050 | 0.2009 | -0.0041 |

## Latency

| Stage | CrossFuse mean | PACE mean |
|---|---:|---:|
| cross_attention_block_ms | 49.206 | N/A |
| cuda_peak_allocated_mb | 227.720 | 547.852 |
| detection_nms_ms | 7.364 | 12.866 |
| encoder_ms | 5.050 | N/A |
| encoder_quality_gate_ms | N/A | 37.622 |
| end_to_end_ms | 68.840 | 60.029 |
| fps_from_mean_latency | 14.527 | 16.659 |
| fusion_ms | 56.426 | N/A |
| selective_fusion_ms | N/A | 9.541 |

## Parameter counts

| Component | CrossFuse | PACE |
|---|---:|---:|
| detector | 2,590,527 | 2,590,527 |
| fusion_without_detector | 20,801,123 | N/A |
| ir_encoder | 1,254,529 | 1,254,529 |
| ir_quality_estimator | N/A | 94,593 |
| local_fusion | N/A | 25,900,708 |
| reliability_gate | N/A | 337 |
| rgb_encoder | 1,254,529 | 1,254,529 |
| rgb_quality_estimator | N/A | 94,593 |
| total | 23,391,650 | N/A |
| total_including_detector | N/A | 31,189,816 |

## Generated charts

- `detection_precision.png`
- `detection_recall.png`
- `detection_f1.png`
- `detection_map50.png`
- `detection_map50_95.png`
- `detection_overview.png`
- `per_class_ap.png`
- `per_class_recall.png`
- `subset_map50.png`
- `subset_map50_95.png`
- `subset_precision.png`
- `subset_recall.png`
- `subset_f1.png`
- `degradation_degradation_type_precision.png`
- `degradation_degradation_type_recall.png`
- `degradation_degradation_type_f1.png`
- `degradation_modality_precision.png`
- `degradation_modality_recall.png`
- `degradation_modality_f1.png`
- `degradation_severity_precision.png`
- `degradation_severity_recall.png`
- `degradation_severity_f1.png`
- `latency_mean.png`
- `latency_median.png`
- `latency_p95.png`
- `latency_distribution.png`
- `throughput_fps.png`
- `memory_comparison.png`
- `parameter_counts.png`
- `fusion_quality_mean.png`
- `fusion_quality_median.png`
- `gate_activity.png`
- `fused_mean_intensity.png`
- `fused_max_intensity.png`
- `sample_prediction_counts.png`
- `sample_max_confidence.png`
- `sample_ground_truth_counts.png`
- `sample_latency_scatter.png`
- `sample_prediction_scatter.png`

## Visualization gallery

- Matched side-by-side panels: 12
- Matched standalone fused images: 0
- Contact sheets: 2
- Pair images: `/home/nvidia/PACE/evaluation/comparisons/test_100/visualization_pairs`

## Important interpretation notes

- Missing metrics are reported as N/A; they are not treated as zero.
- A chart is generated only when the corresponding metric exists in at least one run.
- Per-sample comparisons use the intersection of sample IDs present in both `per_sample.csv` files.
- Visualization pairs are matched by PNG filename stem. If the two evaluators use different filenames, those samples will not appear in paired galleries.
- CUDA peak memory is compared only if the evaluation summaries recorded it. This script cannot reconstruct peak GPU memory after the evaluation has finished.
- Parameter counts are model parameter counts, not runtime memory usage.
- Latency numbers should only be compared directly when both runs used the same hardware, device, input size, warmup policy, and timing methodology.
