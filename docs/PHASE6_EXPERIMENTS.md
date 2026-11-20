# Phase 6: experiments and artifacts

## Batch 1 — read-only inventory

Run `python tools/inventory_experiments.py` to create deterministic JSON/CSV inventories and a summary in `reports/refactor/phase6_experiment_inventory/`.

- No files are moved, deleted, or rewritten by the inventory scanner.
- Scans scripts and research-related artifact formats, with SHA-256 for files up to 32 MiB by default. Large files are listed with an explicit unhashed status. Use `--include-large` for a complete checksum inventory.
- Known bulky dataset directories are intentionally excluded. Results and comparison directories are scanned. File paths are relative to repository root.
- Inspect the inventory before proposing any reorganizations. The existing `evaluation/comparisons/both_baselines_vs_pace_test_set/` and partner's RGB sensor-noise findings are important research evidence.
- Some local-only scripts and artifacts may not be on GitHub; inventory is run locally to capture them.

## Batches 2 and 3

Batch 2: checksum/manifest reproducibility and checkpoint provenance, without altering checkpoint contents.

Batch 3: research-results integrity audit, reference mapping, and safe cleanup plan. No deletion without approval.

## Known limitation

The original condition-specific CSVs, plots, and configuration from the partner have not yet been verified in this repository. Do not claim they are preserved until inspected.
