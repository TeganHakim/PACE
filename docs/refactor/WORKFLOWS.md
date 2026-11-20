# PACE reproducible workflows

Run from the repository root with the project environment active.

## Data
- Original dataset: `VisDrone-DroneVehicle/`.
- Generated datasets: `processed_dataset/` and `processed_dataset_realistic_degraded_v2/`.
- Original generation instructions: `prototyping/quality_estimator/synthetic_data_degradation/README.md`.
- Realistic degraded test dataset description and download link: root `README.md`.
- Keep random seeds, preprocessing/cropping and annotation coordinate systems fixed.

## Training
The refactor preserves original training implementations and pretrained checkpoints.

```bash
python -m pace.training.quality_estimator --help
python -m pace.training.reliability_gate --help
```

Training can require datasets and substantial resources. These help commands inspect entrypoints; they do not retrain.

## End-to-end evaluation

```bash
python evaluation/eval_metrics.py \
  --dataset-root processed_dataset_realistic_degraded_v2 \
  --original-root VisDrone-DroneVehicle \
  --max-samples 3 --device mps --visualizations 0 --warmup 2 --seed 42 \
  --output evaluation/results/phase7_smoke
```

Use `--device cuda` only on a CUDA-enabled system, or `--device cpu` for CPU testing. For a full run use `--max-samples 0` with adequate disk, memory and time.

## Scientific regression
Compare against a **locally retained** Phase 4 reference:

```bash
python evaluation/verify_regression.py \
  --reference evaluation/results/phase4_inference_validation_v2 \
  --candidate evaluation/results/phase7_smoke
```

The checker intentionally excludes nondeterministic latency values. Cross-device numerical agreement may need a looser tolerance.

## Integrity and tests

```bash
python -m pytest -q tests/
python tools/verify_research_artifacts.py --verify reports/refactor/phase6_artifact_manifest.json --strict
python tools/audit_research_outputs.py
```

If the manifest fails after an intentional source edit, investigate the difference before creating a replacement baseline. Do not silently overwrite the recorded provenance.
