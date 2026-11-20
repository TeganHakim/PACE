# Phase 5 evaluation validation

## What is preserved
The PACE, CrossFuse and Addition evaluation pipelines and their inference/metrics calculations remain the same. Phase 5 replaced dynamic source-code injection in the PACE and CrossFuse E2E wrappers with explicit opt-in CLI flags. Existing condition-analysis logic and full-test comparison reports are retained.

## Mac acceptance (completed by user before final bundle)
- 100 tests passed after explicit-hook migration.
- 3-image clean/degraded E2E evaluation completed on MPS.
- All six method/split summary scientific metrics matched the Phase 4 reference at absolute tolerance 1e-10, after excluding non-deterministic latency values.

## Additional final-bundle tests
Condition analysis: class-aware IoU matching, confidence filtering, one-to-one assignment, no-ground-truth behavior, duplicate sample rejection, bootstrap determinism. Jetson preflight: CUDA detection and required checkpoint paths.

## Important limitations
- Three-image smoke tests verify refactor parity, **not** scientific superiority.
- The full 8,980-image evaluation report and partner's condition-specific RGB sensor-noise findings must be retained. Partner's original condition-analysis CSVs/figures/config are not yet confirmed present in the repository.
- Jetson is **not validated** until actual CUDA inference is run there. A Mac MPS pass is not a CUDA pass.
- Do not compare timing metrics for exact equality. Record latency/FPS separately on each device.
- Preserve `*.phase5.bak` files until the team has committed and independently verified the migration. Do not commit backups.

## Jetson acceptance commands (run on the actual Jetson, with correct JetPack-specific dependencies)
```bash
python evaluation/jetson_preflight.py --require-cuda --output reports/refactor/jetson_preflight.json
python -m pytest -q tests/
python evaluation/eval_metrics.py --dataset-root processed_dataset_realistic_degraded_v2 --original-root VisDrone-DroneVehicle --max-samples 3 --device cuda --visualizations 0 --warmup 2 --seed 42 --output evaluation/results/jetson_phase5_smoke
```
Compare scientific metrics allowing reasonable hardware numeric tolerance; do not expect identical latency. Run the full dataset after the smoke passes.
