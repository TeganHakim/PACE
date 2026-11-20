
# PACE Architecture (Post-Refactor)

PACE = **Patch-Adaptive Cross-Attention mechanism for Efficient sensor fusion**.

PACE is a multimodal RGB–infrared (IR) image-fusion framework designed to adaptively combine information from two sensing modalities based on their estimated local quality and reliability.

The system uses spatial quality estimation, a reliability gate, and patch-selective cross-attention fusion to produce images for downstream object detection.

This document describes the repository organization, active inference pipeline, major components, and research-preservation principles following the codebase refactor.

---

## 1. Repository Structure

The repository separates the PACE implementation, baseline methods, experimental code, evaluation infrastructure, and research artifacts.

The tree below highlights important directories and files rather than listing every generated output, checkpoint, or dataset image.

```text
PACE/
│
├── README.md                         # Main project documentation
├── LICENSE
├── requirements.txt                  # Python dependencies
│
├── pace/                             # PACE implementation and interfaces
│   ├── __init__.py
│   ├── dataset.py                    # Dataset loading and preprocessing
│   ├── evaluate.py                   # Primary PACE evaluator
│   ├── evaluate_original.py          # Original evaluation variant
│   ├── evaluate_interpolate_encoder.py
│   ├── evaluate_resize_only.py
│   ├── visualize_one_image_pace.py   # Single-example visualization
│   │
│   ├── data/                         # Shared data-processing utilities
│   │   ├── degradation.py
│   │   ├── lighting.py
│   │   └── quality_labels.py
│   │
│   ├── models/                       # Stable model import interfaces
│   │   ├── fusion.py
│   │   ├── gate.py
│   │   └── quality.py
│   │
│   └── training/                     # Training entrypoints and contracts
│       ├── _legacy.py
│       ├── contract_audit.py
│       ├── quality_estimator.py
│       └── reliability_gate.py
│
├── baseline/                         # Reference fusion methods
│   ├── Addition/
│   │   ├── fuse.py                   # Fixed-weight image fusion
│   │   └── evaluate.py
│   │
│   └── CrossFuse/
│       ├── evaluate.py
│       ├── train_autoencoder.py
│       ├── train_conv_trans.py
│       ├── network/                  # CrossFuse neural architectures
│       ├── models/                   # Trained model artifacts
│       └── tools/
│
├── evaluation/                       # Shared evaluation framework
│   ├── eval_metrics.py               # Multi-method evaluation orchestrator
│   ├── pace_e2e.py                   # PACE end-to-end wrapper
│   ├── crossfuse_e2e.py              # CrossFuse end-to-end wrapper
│   ├── addition_e2e.py               # Addition end-to-end wrapper
│   ├── common.py                     # Shared evaluation utilities
│   ├── fusion_metrics.py             # Fusion-quality measurements
│   ├── fusion_diagnostics.py
│   ├── compare.py
│   ├── compare_three.py
│   ├── condition_analysis.py         # Condition-specific performance analysis
│   ├── verify_regression.py          # Refactor output comparison
│   ├── jetson_preflight.py           # Hardware and checkpoint checks
│   ├── visualize_results.py
│   ├── comparisons/                  # Historical research comparisons
│   └── results/                      # Generated evaluation outputs
│
├── prototyping/                      # Original research implementations
│   ├── encoder/                      # Feature encoder experiments
│   ├── quality_estimator/            # Spatial quality-estimation models
│   ├── reliability_gate/             # Gate training and diagnostics
│   ├── selective_fusion/             # LocalCrossFuse implementation
│   │   ├── inference.py              # Active model loading and inference
│   │   ├── local_crossfuse.py
│   │   └── validate_gated_local.py
│   ├── synthetic_data_degradation/   # Dataset degradation generation
│   └── dataset_tools.py
│
├── tests/                            # Automated tests and regressions
│   ├── test_data_module_parity.py
│   ├── test_dataset_preprocessing.py
│   ├── test_model_import_compatibility.py
│   ├── test_pretrained_model_compatibility.py
│   ├── test_reliability_gate.py
│   ├── test_selective_fusion.py
│   ├── test_training_contract.py
│   ├── test_evaluation_regression.py
│   ├── test_condition_analysis_contract.py
│   └── test_phase8_scientific_audit.py
│
├── tools/                            # Maintenance and validation utilities
│   ├── inventory_experiments.py
│   ├── audit_research_outputs.py
│   ├── verify_research_artifacts.py
│   ├── phase8_scientific_audit.py
│   ├── phase8_cleanup_audit.py
│   ├── phase8_mac_acceptance.py
│   └── phase8_jetson_handoff.py
│
├── docs/                             # Technical documentation
│   └── refactor/
│       ├── ARCHITECTURE.md
│       ├── WORKFLOWS.md
│       ├── ENVIRONMENTS.md
│       └── PHASE8_ACCEPTANCE.md
│
├── reports/                          # Project reports and audit artifacts
│   ├── project_proposal/
│   ├── progress_notes.md
│   └── refactor/
│
├── VisDrone-DroneVehicle/            # Original paired RGB/IR dataset
├── processed_dataset/                # Processed dataset
├── processed_dataset_realistic_degraded_v2/
│                                       # Realistically degraded evaluation data
│
├── build_realistic_degraded_test.py  # Realistic test-set generation
├── download_dataset.py               # Dataset download utility
└── preview_addition.py               # Addition baseline preview
```

### Directory responsibilities

| Directory | Responsibility |
|---|---|
| `pace/` | Primary PACE interfaces, evaluation entrypoints, shared data utilities, and model compatibility imports |
| `baseline/` | Reference fusion methods used for comparative experiments |
| `evaluation/` | Consistent end-to-end evaluation, metrics, condition analysis, and visualization |
| `prototyping/` | Original research implementations, training algorithms, and experimental utilities |
| `tests/` | Unit tests, compatibility checks, and behavioral regression tests |
| `tools/` | Experiment inventories, artifact verification, and hardware validation |
| `docs/` | Architecture, workflow, environment, and migration documentation |
| `reports/` | Research reports, project materials, and refactor audit evidence |

**Important:** `prototyping/` is not an obsolete directory. Several active model implementations and training algorithms remain there. The refactor introduced stable interfaces without rewriting the underlying research implementations.

---

## 2. Active Inference Pipeline

The active PACE inference path is:

```text
              Paired RGB and IR Images
                         |
                         v
               Dataset Preprocessing
                         |
               +---------+---------+
               |                   |
               v                   v
           RGB Encoder         IR Encoder
               |                   |
               v                   v
          RGB Features         IR Features
               |                   |
               v                   v
       RGB Quality Estimator  IR Quality Estimator
               |                   |
               v                   v
        RGB Quality Map       IR Quality Map
               |                   |
               +---------+---------+
                         |
                         v
                  Reliability Gate
                         |
                         v
               Patch Routing Decision
                         |
                         v
              Patch-Selective LocalCrossFuse
                         |
                         v
                    Fused Image
                         |
                         v
                 YOLO11n Detector
                         |
                         v
                Detection Predictions
```

### 2.1 Input preprocessing

PACE operates on paired RGB and infrared images representing the same scene.

Dataset loading and preprocessing are handled through `pace/dataset.py` and shared utilities under `pace/data/`.

The evaluation pipeline is responsible for providing corresponding inputs to each method so comparisons use matched samples.

### 2.2 Feature encoding

The RGB and IR modalities are processed by pretrained, frozen feature encoders.

These encoders produce feature representations used by the spatial quality estimators.

The encoder weights are retained during quality-estimator and reliability-gate inference.

### 2.3 Spatial quality estimation

Separate quality estimators predict local quality information for the RGB and IR modalities.

The quality-estimation pipeline is associated with:

- `prototyping/quality_estimator/`
- `pace/models/quality.py`
- `pace/training/quality_estimator.py`

The project uses an 8 × 8 patch grid for patch-level quality and fusion decisions.

The quality maps provide information about which modality is more reliable in different spatial regions.

### 2.4 Reliability gate

The reliability gate uses the estimated quality information to determine where selective fusion should be applied.

Relevant files include:

- `prototyping/reliability_gate/reliability_gate.py`
- `pace/models/gate.py`
- `pace/training/reliability_gate.py`

The gate supports selective fusion rather than requiring every spatial region to undergo the same fusion operation.

### 2.5 Patch-selective LocalCrossFuse

The fusion stage uses local cross-attention to combine information from RGB and IR inputs in selected spatial regions.

The active implementation remains under:

- `prototyping/selective_fusion/inference.py`
- `prototyping/selective_fusion/local_crossfuse.py`

The `pace/models/fusion.py` module provides a stable import interface.

The resulting fused image is reconstructed for downstream evaluation.

### 2.6 Object detection

The fused image is evaluated using a YOLO11n object detector.

Detection performance provides a downstream task-based measure of fusion usefulness.

Fusion quality and object detection are evaluated separately because improvements in image-level fusion metrics do not necessarily imply improvements in detection accuracy.

---

## 3. Model and Training Organization

The refactor separates stable interfaces from the original research implementations.

### Model interfaces

`pace/models/` exposes compatibility imports for the existing model classes.

These modules help centralize imports without changing the underlying model architectures or checkpoint formats.

### Training interfaces

`pace/training/` provides entrypoints for the existing quality-estimator and reliability-gate training workflows.

The original training algorithms remain in their research directories.

| Component | Primary research implementation | Refactored interface |
|---|---|---|
| Quality estimation | `prototyping/quality_estimator/` | `pace/models/quality.py`, `pace/training/quality_estimator.py` |
| Reliability gate | `prototyping/reliability_gate/` | `pace/models/gate.py`, `pace/training/reliability_gate.py` |
| Selective fusion | `prototyping/selective_fusion/` | `pace/models/fusion.py` |
| Dataset utilities | `prototyping/synthetic_data_degradation/` | `pace/data/` |

The refactor is intended to preserve pretrained checkpoint compatibility and existing scientific behavior.

---

## 4. Evaluation Architecture

The shared evaluation framework is located under `evaluation/`.

### Evaluated methods

The principal comparisons use three fusion methods:

| Method | Description |
|---|---|
| PACE | Quality-guided, reliability-gated, patch-selective fusion |
| CrossFuse | Reference cross-attention-based fusion method |
| Addition | Fixed 50/50 RGB–IR image addition |

### Evaluation workflow

```text
              Matched RGB / IR Test Pairs
                         |
                         v
              evaluation/eval_metrics.py
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
           PACE       CrossFuse    Addition
             |           |           |
             v           v           v
          Fused       Fused       Fused
          Images      Images      Images
             |           |           |
             +-----------+-----------+
                         |
                         v
                Detection Evaluation
                         |
                         v
                 Fusion Metrics
                         |
                         v
              Comparison Reports / Figures
```

`evaluation/eval_metrics.py` orchestrates evaluation of all three methods on matched samples, including clean and degraded conditions.

The method-specific wrappers are:

- `evaluation/pace_e2e.py`
- `evaluation/crossfuse_e2e.py`
- `evaluation/addition_e2e.py`

### Condition-specific analysis

`evaluation/condition_analysis.py` performs post-hoc analysis to identify conditions under which PACE improves over the Addition baseline.

This analysis is separate from the primary end-to-end evaluation.

Condition-specific findings should be interpreted using the exact dataset, metrics, configurations, and analysis outputs used to produce them.

### Historical results

Historical comparison reports are preserved under:

`evaluation/comparisons/`

The full-test comparison report is located at:

`evaluation/comparisons/both_baselines_vs_pace_test_set/comparison_report.md`

Generated evaluation runs are stored under:

`evaluation/results/`

These directories should not be indiscriminately deleted, since they may contain experimental evidence required for reproducing reported findings.

---

## 5. Data Organization

The repository contains several dataset directories serving different purposes.

| Path | Purpose |
|---|---|
| `VisDrone-DroneVehicle/` | Original paired RGB/IR source dataset |
| `processed_dataset/` | Processed dataset used in the project |
| `processed_dataset_realistic_degraded_v2/` | Realistically degraded dataset for evaluation |
| `build_realistic_degraded_test.py` | Script for constructing the realistic degraded test dataset |
| `pace/data/` | Shared degradation, lighting, and quality-label utilities |

Synthetic degradation and quality-label generation were originally implemented under:

`prototyping/synthetic_data_degradation/`

The shared data utilities introduced by the refactor preserve compatibility with the original implementations.

Dataset construction should remain reproducible through explicit preprocessing configurations and random seeds.

---

## 6. Testing and Validation

Automated tests are located under `tests/`.

They cover:

- Data preprocessing and quality-label compatibility
- Model import and pretrained checkpoint compatibility
- Reliability-gate and selective-fusion behavior
- Training interfaces and contracts
- Evaluation entrypoints and regression behavior
- Research artifact and experiment inventory checks
- Scientific audit and acceptance utilities

Run the test suite from the repository root:

```bash
python -m pytest -q tests/
```

### Refactor regression validation

The refactor was validated on an Apple Silicon Mac using PyTorch MPS.

The final Mac-side results were:

| Validation | Result |
|---|---|
| Automated tests | 147 passed |
| End-to-end regression | 6/6 method/split evaluations matched |
| Numeric tolerance | 1e-10 |
| Checkpoint integrity | Passed |
| Mac environment preflight | Passed |
| Jetson CUDA validation | Pending |
| Full 8,980-image rerun | Intentionally skipped |

The six regression comparisons cover PACE, CrossFuse, and Addition on clean and degraded samples.

Regression comparison can be reproduced using:

```bash
python evaluation/verify_regression.py \
  --reference evaluation/results/phase5_explicit_hooks \
  --candidate evaluation/results/phase8_mac_smoke
```

Latency and throughput are excluded from the numeric-equivalence assertion because they depend on execution hardware and runtime conditions.

The successful three-sample smoke test demonstrates preservation of measured behavior on those samples, not general model superiority.

---

## 7. Jetson CUDA Validation

The final hardware-specific validation is intended to run on the NVIDIA Jetson environment.

The Jetson handoff utility is:

`tools/phase8_jetson_handoff.py`

A teammate can check out the `reorganization` branch and run:

```bash
git fetch origin
git switch reorganization
git pull --ff-only origin reorganization

conda activate pace
python tools/phase8_jetson_handoff.py
```

The handoff workflow checks the CUDA environment and required checkpoints, runs the automated tests, and performs a small end-to-end evaluation.

A successful Jetson run provides additional hardware-specific confidence but does not replace full scientific replication.

See `docs/refactor/PHASE8_ACCEPTANCE.md` for further details.

---

## 8. Research Integrity and Reproducibility

The refactor prioritizes preserving the scientific behavior of the original implementation.

### Preservation principles

1. Do not change model architectures, checkpoint contents, or training algorithms solely for repository organization.
2. Preserve original experimental implementations and relevant historical outputs.
3. Keep dataset preprocessing and sample selection consistent when comparing methods.
4. Separate software regression validation from claims about scientific performance.
5. Retain experiment configurations, metrics, and artifacts needed to reproduce reported results.

### Interpretation of results

The three-sample smoke test must not be treated as evidence of model superiority.

The historical full-test comparison report is available under:

`evaluation/comparisons/both_baselines_vs_pace_test_set/`

Condition-specific RGB sensor-noise findings require the original analysis CSVs, figures, and exact configuration for independent verification.

The successful Mac regression establishes that the refactored implementation reproduced the reference outputs for the tested samples within the specified tolerance.

It does not establish that all historical full-dataset results have been independently reproduced.

---

## 9. Related Documentation

| Document | Purpose |
|---|---|
| `README.md` | Project overview and entrypoint |
| `docs/refactor/ARCHITECTURE.md` | Repository structure and component architecture |
| `docs/refactor/WORKFLOWS.md` | Dataset, training, and evaluation workflows |
| `docs/refactor/ENVIRONMENTS.md` | Environment setup and dependencies |
| `docs/refactor/PHASE8_ACCEPTANCE.md` | Final validation and Jetson handoff |
| `evaluation/README_EVALUATION.md` | Evaluation documentation |
| `evaluation/READ_CONDITION_ANALYSIS.md` | Condition-analysis workflow |

---

## 10. Development Guidance

For future development:

- Add shared preprocessing functionality under `pace/data/`.
- Use `pace/models/` for stable model imports and interfaces.
- Keep training entrypoints under `pace/training/`, preserving original research implementations as appropriate.
- Use `evaluation/` for cross-method comparisons and scientific evaluation.
- Add automated tests when modifying model interfaces, preprocessing, fusion, or evaluation behavior.
- Preserve checkpoints, experiment outputs, and configurations required for reproducibility.
- Avoid modifying historical evaluation results in place.

The intended architecture is a modular research codebase that supports continued experimentation while preserving the original PACE implementation and its scientific evidence.
