# Phase 8 acceptance and partner handoff

All scripts are additive; no trained weights, scientific algorithms, or prior research outputs are modified.

## 8.1 Scientific evidence
`python tools/phase8_scientific_audit.py --strict`

The prior **8,980 matched-sample** report records CrossFuse/PACE/Addition mAP@50 of **0.4712 / 0.4544 / 0.4417**. It does **not** demonstrate PACE beats CrossFuse overall. The condition-specific RGB sensor-noise result is a separate claim requiring direct CSV/figure inspection or a fresh `evaluation/condition_analysis.py` run. A successful audit only checks the historical report is present and parseable.

## 8.2 Cleanup
`python tools/phase8_cleanup_audit.py`

Read-only. Inspect the inventory before deleting anything. Never auto-delete datasets, checkpoints, experimental outputs, or provenance files.

## 8.3 Mac
```bash
python tools/phase8_mac_acceptance.py --e2e --reference evaluation/results/phase5_explicit_hooks
```
If the reference is elsewhere, use its actual path. Omit `--reference` if unavailable, but then no numeric regression comparison is performed. For the full dataset (long-running), use `--full` without the 3-sample reference. Tests, artifact manifest verification and CUDA-neutral preflight run first.

## 8.4 Jetson Partner Handoff

Run all commands from the PACE repository root on the NVIDIA Jetson.

### Prerequisites

Before starting, ensure the Jetson has:

- A working JetPack-compatible CUDA PyTorch environment
- The original `VisDrone-DroneVehicle/` dataset
- The `processed_dataset_realistic_degraded_v2/` dataset
- All four pretrained model checkpoints

### Stage 1: CUDA Compatibility and Smoke Test

```bash
git fetch origin
git switch reorganization
git pull --ff-only origin reorganization

conda activate pace
python tools/phase8_jetson_handoff.py
```

This performs:

1. CUDA environment and checkpoint preflight
2. Automated tests
3. Three-sample end-to-end evaluation of PACE, CrossFuse, and Addition

Outputs:

`evaluation/results/phase8_jetson_smoke/`

This stage verifies basic execution compatibility, not full-dataset performance.

### Stage 2: Full Evaluation and Updated Figures

After Stage 1 passes:

```bash
python evaluation/eval_metrics.py \
  --dataset-root processed_dataset_realistic_degraded_v2 \
  --original-root VisDrone-DroneVehicle \
  --max-samples 0 \
  --device cuda \
  --visualizations 12 \
  --warmup 5 \
  --seed 42 \
  --output evaluation/results/phase8_jetson_full
```

This evaluates all available test pairs in the selected dataset using:

- PACE
- CrossFuse
- 50/50 Addition

The evaluation includes clean and degraded conditions and generates updated metrics, comparison figures, and representative visualizations.

`--max-samples 0` means all available samples. Verify the actual sample count rather than assuming exactly 8,980.

Do not overwrite historical evaluation results.

### Stage 3: Condition-Specific Analysis

After the full evaluation finishes:

```bash
python evaluation/condition_analysis.py \
  --run evaluation/results/phase8_jetson_full
```

This generates condition-specific comparisons and supporting CSVs and figures.

Inspect these results before claiming that PACE outperforms Addition under particular degradation conditions, including RGB sensor noise.

### Stage 4: Share Results

Please share:

- Complete terminal logs
- CUDA preflight results
- Three-sample smoke-test summary
- Full evaluation metrics and figures
- Condition-analysis outputs
- Git commit SHA and JetPack/PyTorch versions

Keep the new results separate from historical experiments.

### Scientific Interpretation

The smoke test verifies basic execution compatibility.

The full evaluation measures performance on the realistically degraded dataset. It does not automatically reproduce historical results obtained using a different dataset or experimental configuration.

Jetson validation remains pending until these tests are successfully executed on the physical device.
