# Phase 3: Incremental migration

## Step 1: quality labels (additive)

The original `prototyping/synthetic_data_degradation/quality_labels.py` is
preserved verbatim. `pace/data/quality_labels.py` contains the same source and
provides a future package import path. The existing dataset builder continues
to use the original module; no production caller has been switched yet.

`tests/test_quality_labels_parity.py` checks source equality and output parity.
Before switching imports, rerun the existing 44 regression tests and the new
parity suite. Do not delete the legacy module until all callers have migrated.

No checkpoints, datasets, results, or condition-analysis artifacts are modified.
