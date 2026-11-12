# CrossFuse vs. PACE — paired evaluation

Compared **8980** matched samples.

## Detection performance

| Metric | CrossFuse | PACE | Difference (PACE − CrossFuse) |
|---|---:|---:|---:|
| Precision | 0.8745 | 0.8652 | -0.0093 |
| Recall | 0.6903 | 0.6755 | -0.0148 |
| F1 | 0.7716 | 0.7586 | -0.0129 |
| mAP@50 | 0.4712 | 0.4540 | -0.0172 |
| mAP@50:95 | 0.2936 | 0.2775 | -0.0161 |

Detection metrics above are evaluator-reported **dataset-level point estimates**. No p-values or confidence intervals are assigned to mAP without recomputing mAP over paired bootstrap resamples.

## Latency and memory

Mean paired end-to-end latency difference (PACE − CrossFuse): **-14.070 ms** (n=8980).
Paired percentile bootstrap 95% CI: [-14.250, -13.899] ms.
Recorded peak GPU memory: CrossFuse 321.5 MB; PACE 674.4 MB. This is **one run-level measurement per model**, not an inferential estimate.

## Figures: provenance, interpretation, and purpose

### 1. `01_detection.png`

**Goal:** Compare dataset-level detection accuracy on matched samples.

**Data source(s):** `baseline/summary.json: detection`, `pace/summary.json: detection`.

**How to read / calculation:** Grouped bars show evaluator-reported dataset-level metrics; printed values are point estimates, not confidence intervals.

### 2. `02_degradation.png`

**Goal:** Compare detection performance on clean and degraded images.

**Data source(s):** `baseline/summary.json: subsets`, `pace/summary.json: subsets`.

**How to read / calculation:** Two condition panels with grouped bars for F1 and mAP@50; values are descriptive subset-level estimates.

### 3. `03_latency.png`

**Goal:** Compare per-image latency distributions on matched samples.

**Data source(s):** `baseline/per_sample.csv: sample_id,end_to_end_ms`, `pace/per_sample.csv: sample_id,end_to_end_ms`.

**How to read / calculation:** Boxplots show median, IQR, whiskers and outliers; diamonds mark means. Positive PACE-minus-CrossFuse difference means PACE is slower. Paired bootstrap is reported separately.

### 4. `04_stage_latency.png`

**Goal:** Compare overall stage costs and expose fusion savings versus gate overhead.

**Data source(s):** `baseline/latency.json: encoder_ms,fusion_ms,detection_nms_ms`, `pace/latency.json: encoder_quality_gate_ms,selective_fusion_ms,detection_nms_ms`.

**How to read / calculation:** Stacked mean stage times. The first segment is not functionally identical: PACE includes quality estimation and gating. Stage sums can differ slightly from separately timed end-to-end means.

### 5. `05_memory.png`

**Goal:** Compare observed GPU allocation overhead.

**Data source(s):** `baseline/latency.json: cuda_peak_allocated_mb`, `pace/latency.json: cuda_peak_allocated_mb`.

**How to read / calculation:** Single run-level peak per model; descriptive only, not a sample-level distribution or confidence interval.

### 6. `06_gate_latency.png`

**Goal:** Show how observed runtime varies with the fraction of patches fused.

**Data source(s):** `pace/per_sample.csv: active_fraction,end_to_end_ms`.

**How to read / calculation:** Nonoverlapping gate-activity bins, including a separate zero-activation bin. Bars are per-bin means and labels give sample counts; descriptive only.

### 7. `07_tradeoff.png`

**Goal:** Quantify whether fusion savings compensate for quality estimation and gating.

**Data source(s):** `baseline/latency.json: encoder_ms,fusion_ms`, `pace/latency.json: encoder_quality_gate_ms,selective_fusion_ms`.

**How to read / calculation:** Fusion time saved = CrossFuse fusion mean − PACE fusion mean. Additional encoder/gate cost = PACE encoder-quality-gate mean − CrossFuse encoder mean. Positive overhead offsets positive savings.

### 8. `visualization_pairs/`

**Goal:** Inspect differences in fused imagery and detection overlays for the same images.

**Data source(s):** `baseline/visualizations/<sample_id>.png`, `pace/visualizations/<sample_id>.png`.

**How to read / calculation:** Filename-matched qualitative side-by-side panels; not quantitative evidence of improvement.

## Methodology and limitations

- Both evaluators are run sequentially with identical split, requested image count, confidence, NMS IoU, seed, device selection and warmup arguments. They retain their own existing CLI and output schemas.
- Exact sample IDs must match across the two runs. A mismatch is an error rather than an invitation to silently compare different subsets.
- Latency confidence intervals resample **paired images**, preserving correspondence. They quantify sample variability conditional on this device/run, not run-to-run GPU variability.
- Inference timing may be affected by thermal state, GPU contention, caching and evaluator-specific instrumentation; sequential execution is not randomized.
- The clean/degraded subset counts and per-subset detection metrics are descriptive; sparse categories should not be interpreted as robust effects.
- GPU peak memory is measured at run level; a single observation per pipeline cannot support a statistical significance claim.
- Model parameters and memory are not interchangeable. Memory includes loaded models and temporary allocations.
- Gate activity is observational; no causal interpretation is warranted.

## Inputs

- CrossFuse: `/home/nvidia/PACE/baseline/CrossFuse/evaluation/baseline_epoch_28_test_8980`
- PACE: `/home/nvidia/PACE/pace/evaluation/pace_test_8980_interpolate_encoder`
