# PACE architecture (post-refactor)

PACE = Patch-Adaptive Cross-Attention mechanism for Efficient sensor fusion.

## Active inference path

RGB/IR images -> frozen feature encoders -> RGB/IR spatial quality estimators -> reliability gate -> patch-selective LocalCrossFuse -> fused image -> YOLO11n detection.

- `pace/models/`: compatibility imports for the existing model classes; original implementation remains under `prototyping/`.
- `pace/data/`: shared dataset quality labels and degradation utilities.
- `prototyping/selective_fusion/inference.py`: model loading and inference functions.
- `pace/evaluate.py`: held-out PACE evaluator with explicit `--paired-jpg-samples` and `--save-fused` flags.
- `baseline/CrossFuse/evaluate.py`: CrossFuse baseline evaluator.
- `evaluation/eval_metrics.py`: orchestrates PACE, CrossFuse and 50/50 Addition on the same samples, including clean/degraded comparisons.
- `evaluation/condition_analysis.py`: post-hoc analysis of conditions under which PACE improves over Addition.
- `pace/training/`: entrypoints for original quality estimator and reliability gate training; existing training algorithms retained.

## Research integrity

Do not treat the 3-sample smoke test as evidence of model superiority. The full-test comparison report is under `evaluation/comparisons/both_baselines_vs_pace_test_set/`. Condition-specific RGB sensor-noise findings require their original analysis CSVs/figures and exact configuration for independent verification.

Refactor regression: 6 method/split scientific summaries matched the Phase 4 reference at tolerance 1e-10 on Mac MPS, excluding latency. Jetson CUDA execution is not yet verified.
