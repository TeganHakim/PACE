# PACE vs. 50/50 Addition: Condition Analysis

**Goal:** Find *when* PACE performs better than simple averaging—for example, when RGB is noisy but IR is sharp—using actual detection results rather than counting extra edges.

## 1. Run it

Place `condition_analysis.py` in `PACE/evaluation/`. First run the existing `evaluation/eval_metrics.py` evaluation on the dataset (ideally **hundreds of images**, with `--original-root` so clean references are saved).

From the **PACE repository root**:

```bash
python evaluation/condition_analysis.py \
  --run evaluation/results/YOUR_RUN_FOLDER \
  --min-leaf 20 \
  --examples 8
```

Replace `YOUR_RUN_FOLDER` with your completed evaluation folder. The script **does not rerun models**; it reads saved PACE and Addition predictions and fused images. It uses the dataset location saved in the run's `config.json`.

If you moved the dataset, add `--dataset-root /path/to/processed_dataset_realistic_degraded_v2`.

**Dependencies:** `numpy`, `pandas`, `opencv-python`, `matplotlib`, `scikit-learn`.

## 2. What is being compared?

**Main outcome: detection F1 per image.** Each method's predicted vehicle boxes are matched to ground truth at **IoU ≥ 0.5**, with the same confidence cutoff (normally the one used during evaluation).

| Column | Meaning | Favorable for PACE |
|---|---|---|
| `delta_f1` | PACE detection F1 − Addition detection F1 | **Positive** |
| `delta_recall` | PACE recall − Addition recall | Positive |
| `delta_fp` | PACE false-positive count − Addition false-positive count | **Zero or negative** |
| `delta_tp` | PACE true-positive count − Addition true-positive count | Positive |

**Supporting image-quality check (when clean references exist):** `delta_reference_edge_f1` compares fused edges with edges from the **clean RGB and IR source images**. Positive suggests better source-edge preservation. `delta_unsupported_edge_fraction` and `delta_seam_unsupported_fraction` should ideally be **zero or negative**. These are *proxies*, not proof that an edge is real or hallucinated.

## 3. How conditions are discovered

The script measures each input image's **brightness, contrast, sharpness, gradient strength, and edge density**, along with differences between RGB and IR. It also reads the **known corrupted modality and degradation type** from metadata.

It does **not** manually label images “good RGB” or “bad IR.” A shallow decision tree learns numerical conditions associated with better **detection F1**. It learns on **70%** of the images and evaluates the conditions on a separate **30% holdout** by default.

## 4. Read the outputs

All outputs appear in `<run>/condition_analysis/`.

| File | What to look for |
|---|---|
| `condition_summary.csv` | **Start here.** Mean PACE advantage by RGB/IR corruption and noise/blur; inspect `n_images`, `delta_f1_mean`, and `delta_fp_mean`. |
| `degradation_groups.png` | Quick plot of mean F1 advantage for each known degradation condition. Above zero favors PACE. |
| `discovered_rules.csv` | Automatically discovered combinations of image characteristics. Look at **`rule`**, **`test_n_scored`**, **`test_delta_f1_mean`**, **`test_delta_fp_mean`**, and **`holdout_positive_ci`**. |
| `condition_tree.png` | Diagram of which measured input features and thresholds define each rule. |
| `per_image_comparison.csv` | Full per-image inputs, detection metrics, and structural metrics; useful for filtering or plotting. |
| `examples/` | Paired input/fusion images for cases where PACE's detection F1 improved **without extra false positives**. |
| `rule_assignments.csv` | Which rule each image belongs to and whether it was in training or holdout. |
| `analysis_config.json` | Settings, image counts, and metric definitions. |

*If there are too few informative images, the script skips the decision tree and rules; descriptive summaries are still produced.*

## 5. What counts as a convincing finding?

For a candidate pattern (e.g., “IR sharper than RGB when RGB is noisy”), check:

1. **Enough images:** `test_n_scored` is not tiny; avoid drawing conclusions from a handful of examples.
2. **Higher detection F1:** `test_delta_f1_mean > 0`.
3. **No false-positive tradeoff:** `test_delta_fp_mean <= 0` is preferable.
4. **Uncertainty:** `holdout_positive_ci = True` means the **bootstrap 95% interval for mean holdout ΔF1** is entirely above zero. False means the evidence is less conclusive—not necessarily that PACE loses.
5. **Visual sanity check:** Inspect examples and, when available, the reference-edge/seam metrics to rule out obvious patch artifacts.

The `pace_win_rate` column is the **fraction of scored images with ΔF1 > 0**; ties are reported separately. For example, a +0.08 mean ΔF1 with a 65% win rate suggests a useful pattern, **but only if the holdout sample size and false-positive results support it**.

**Important:** These are exploratory associations, **not proof** that brightness or sharpness *caused* the improvement. `examples/` intentionally shows favorable images; use the **holdout statistics**, not those images alone, to support claims in your report.