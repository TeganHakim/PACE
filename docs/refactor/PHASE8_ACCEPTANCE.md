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

## 8.4 Jetson partner handoff
After pushing the branch, on Jetson:
```bash
git fetch origin
git switch reorganization
git pull --ff-only origin reorganization
conda activate pace
python tools/phase8_jetson_handoff.py
```
For full 8,980-image evaluation, rerun with `--full`. This script requires a working JetPack-specific CUDA PyTorch environment, the original and processed datasets, and all four pretrained checkpoints. The partner should return the terminal log and `evaluation/results/phase8_jetson_smoke` summary. Actual Jetson CUDA compatibility cannot be certified from Mac.
