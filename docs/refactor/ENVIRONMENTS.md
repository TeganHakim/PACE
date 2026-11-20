# Mac and Jetson environments

## Apple Silicon development (verified environment)

Phase 6 was validated on macOS 26.6 arm64, Python 3.11.17, PyTorch 2.14.1, with MPS available. All 135 tests passed. This is a **record of a working environment**, not a universal package installation recipe.

```bash
conda activate pace
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
python -m pytest -q tests/
python evaluation/jetson_preflight.py
```

The preflight reports `cuda_available: false` on Mac; this is expected. Use `--device mps` for Mac E2E evaluation.

## NVIDIA Jetson (requires physical validation)

Jetson uses ARM64 Linux and NVIDIA's JetPack-specific CUDA, cuDNN and PyTorch packages. **Do not copy the Mac's PyTorch wheel or assume a desktop CUDA wheel works on Jetson.** Determine the installed JetPack/L4T version first and install the matching NVIDIA-supported PyTorch and torchvision builds. Verify Ultralytics, OpenCV, NumPy, and other dependencies against that environment. Do not replace the original research requirements blindly; Jetson compatibility is not yet certified.

```bash
uname -m
cat /etc/nv_tegra_release
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python evaluation/jetson_preflight.py --require-cuda --output reports/refactor/jetson_preflight.json
python -m pytest -q tests/
python evaluation/eval_metrics.py --dataset-root processed_dataset_realistic_degraded_v2 --original-root VisDrone-DroneVehicle --max-samples 3 --device cuda --visualizations 0 --warmup 2 --seed 42 --output evaluation/results/jetson_smoke
```

After configuring the Jetson environment, follow the complete
[Phase 8 Jetson Partner Handoff](PHASE8_ACCEPTANCE.md#84-jetson-partner-handoff).

The validation proceeds in three stages:

1. CUDA compatibility, automated tests, and three-sample smoke test
2. Full-dataset evaluation with updated metrics and figures
3. Condition-specific analysis of PACE versus the baselines

The smoke test alone does not establish full pipeline performance.
Record Jetson latency, throughput, and memory separately from Mac results.

## Common problems

- `ModuleNotFoundError: pace`: launch from repository root; check `PYTHONPATH` and script entrypoints.
- Checkpoint load errors: check paths and model state-dict contracts; use CPU `map_location` when loading portable weights.
- CUDA unavailable on Jetson: confirm JetPack and NVIDIA-specific torch build before debugging PACE.
- MPS unavailable on Mac: verify torch installation and use `--device cpu` if needed.
- Unexpected regression mismatch: verify dataset IDs, seeds, annotations, checkpoints, and preprocessing before interpreting scientific metrics.
