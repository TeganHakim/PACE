# Visualization guide

## `01_detection.png`

**Goal:** Compare dataset-level detection accuracy on matched samples.

**Source(s):** baseline/summary.json: detection, pace/summary.json: detection

**Method:** Grouped bars show evaluator-reported dataset-level metrics; printed values are point estimates, not confidence intervals.

## `02_degradation.png`

**Goal:** Compare detection performance on clean and degraded images.

**Source(s):** baseline/summary.json: subsets, pace/summary.json: subsets

**Method:** Two condition panels with grouped bars for F1 and mAP@50; values are descriptive subset-level estimates.

## `03_latency.png`

**Goal:** Compare per-image latency distributions on matched samples.

**Source(s):** baseline/per_sample.csv: sample_id,end_to_end_ms, pace/per_sample.csv: sample_id,end_to_end_ms

**Method:** Boxplots show median, IQR, whiskers and outliers; diamonds mark means. Positive PACE-minus-CrossFuse difference means PACE is slower. Paired bootstrap is reported separately.

## `04_stage_latency.png`

**Goal:** Compare overall stage costs and expose fusion savings versus gate overhead.

**Source(s):** baseline/latency.json: encoder_ms,fusion_ms,detection_nms_ms, pace/latency.json: encoder_quality_gate_ms,selective_fusion_ms,detection_nms_ms

**Method:** Stacked mean stage times. The first segment is not functionally identical: PACE includes quality estimation and gating. Stage sums can differ slightly from separately timed end-to-end means.

## `05_memory.png`

**Goal:** Compare observed GPU allocation overhead.

**Source(s):** baseline/latency.json: cuda_peak_allocated_mb, pace/latency.json: cuda_peak_allocated_mb

**Method:** Single run-level peak per model; descriptive only, not a sample-level distribution or confidence interval.

## `06_gate_latency.png`

**Goal:** Show how observed runtime varies with the fraction of patches fused.

**Source(s):** pace/per_sample.csv: active_fraction,end_to_end_ms

**Method:** Nonoverlapping gate-activity bins, including a separate zero-activation bin. Bars are per-bin means and labels give sample counts; descriptive only.

## `07_tradeoff.png`

**Goal:** Quantify whether fusion savings compensate for quality estimation and gating.

**Source(s):** baseline/latency.json: encoder_ms,fusion_ms, pace/latency.json: encoder_quality_gate_ms,selective_fusion_ms

**Method:** Fusion time saved = CrossFuse fusion mean − PACE fusion mean. Additional encoder/gate cost = PACE encoder-quality-gate mean − CrossFuse encoder mean. Positive overhead offsets positive savings.

## `visualization_pairs/`

**Goal:** Inspect differences in fused imagery and detection overlays for the same images.

**Source(s):** baseline/visualizations/<sample_id>.png, pace/visualizations/<sample_id>.png

**Method:** Filename-matched qualitative side-by-side panels; not quantitative evidence of improvement.
