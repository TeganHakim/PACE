# CrossFuse vs. PACE vs. Addition

Compared **8980 matched samples**.

| Metric | CrossFuse | PACE | Addition |
|---|---:|---:|---:|
| Precision | 0.8745 | 0.8650 | 0.8753 |
| Recall | 0.6903 | 0.6769 | 0.6526 |
| F1 | 0.7716 | 0.7595 | 0.7477 |
| mAP@50 | 0.4712 | 0.4544 | 0.4417 |
| mAP@50:95 | 0.2936 | 0.2780 | 0.2723 |

## Methodology

- All metrics were loaded from existing evaluation outputs.
- No CrossFuse or PACE inference was performed.
- All three runs must contain identical sample IDs.
- Detection metrics are dataset-level point estimates.
- Latencies are single-run device measurements.
- Gate and tradeoff figures diagnose PACE relative to CrossFuse.

## Figures

- `figures/01_detection.png`
- `figures/02_degradation.png`
- `figures/03_latency.png`
- `figures/04_stage_latency.png`
- `figures/05_memory.png`
- `figures/06_gate_latency.png`
- `figures/07_tradeoff.png`

## Inputs

- CrossFuse: `/home/nvidia/PACE/baseline/CrossFuse/evaluation/baseline_epoch_28_test_8980`
- PACE: `/home/nvidia/PACE/pace/evaluation/pace_test_8980`
- Addition: `/home/nvidia/PACE/baseline/Addition/evaluation/addition_test_all`