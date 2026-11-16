# PACE Evaluation

Comprehensive evaluation of **PACE**, **CrossFuse**, and **50/50 Addition** on the realistically degraded DroneVehicle test dataset.

## 1. Evaluation Files

- **`eval_metrics.py`** — Main entry point. Runs all three pipelines, evaluates clean and degraded images, computes metrics, and saves results.
- **`pace_e2e.py`** — Runs the existing PACE quality-aware, selective-fusion and object-detection pipeline.
- **`crossfuse_e2e.py`** — Runs the existing full CrossFuse fusion and object-detection pipeline.
- **`addition_e2e.py`** — Runs the 50/50 RGB–IR addition baseline and object detection.
- **`common.py`** — Shared utilities for dataset handling, evaluation execution, and output management.
- **`fusion_metrics.py`** — Computes degradation transfer ratio, Edge F1, and Qᴀʙ/ꜰ.
- **`visualize_results.py`** — Generates metric comparisons, preserves the existing seven comparison figures, and creates qualitative visualizations.

## 2. Evaluation Metrics

**System Performance**
- End-to-end inference latency (ms).
- Peak GPU memory (MB).

**Detection Performance**
- mAP@50.
- Recall.
- Matched detection confidence.

**Fusion Quality and Robustness**
- Degradation transfer ratio.
- Edge F1.
- Qᴀʙ/ꜰ (edge preservation).

Results are compared across all three methods, with additional breakdowns by degraded modality (RGB/IR) and degradation type (sensor noise/droplet blur).

## 3. Generated Outputs

Each evaluation creates a timestamped directory under:

`evaluation/results/YYYY-MM-DD_HH-MM-SS/`

**Per-method outputs**
- `pace/`, `crossfuse/`, `addition/` — Separate clean and degraded evaluation results.
- Each method saves fused images, detection predictions, per-sample statistics, latency measurements, and summary metrics.

**Aggregated metrics — `metrics/`**
- `comparison.csv` — Overall comparison of the three methods.
- `system.csv` — Latency and GPU memory.
- `detection.csv` — mAP@50, recall, and matched detection confidence.
- `fusion_quality.csv` — Aggregated fusion quality and robustness metrics.
- `fusion_per_sample.csv` — Per-image fusion quality measurements.

**Visualizations — `figures/`**
- `existing/` — Seven original comparison figures covering detection, latency, memory, and gating behavior.
- `metrics/` — Individual comparisons of the eight evaluation metrics across all three methods.
- `qualitative/` — Side-by-side fusion images, degradation masks, and detection overlays.

**Other outputs**
- `config.json` — Evaluation settings for reproducibility.
- `_dataset_clean/` and `_dataset_degraded/` — Evaluation dataset caches.

## 4. Run Evaluation

Run all commands from the **PACE repository root**.

### Small Test — 3 Examples

Use this command first to verify that all pipelines execute successfully and generate the expected outputs.

```bash
python evaluation/eval_metrics.py \
  --dataset-root processed_dataset_realistic_degraded_v2 \
  --original-root VisDrone-DroneVehicle \
  --max-samples 3 \
  --device cuda \
  --visualizations 3 \
  --warmup 2
```

### Full Evaluation — Entire Test Dataset

Run after the small test succeeds.

```bash
python evaluation/eval_metrics.py \
  --dataset-root processed_dataset_realistic_degraded_v2 \
  --original-root VisDrone-DroneVehicle \
  --max-samples 0 \
  --device cuda \
  --visualizations 12 \
  --warmup 5
```

Setting `--max-samples 0` evaluates all available test pairs.

Both commands run PACE, CrossFuse, and Addition, calculate the evaluation metrics, export CSVs, and generate comparison figures.

**Note:** Full end-to-end evaluation requires the trained model checkpoints and a compatible execution environment, such as the Jetson. The original DroneVehicle images are required for clean-reference robustness metrics.