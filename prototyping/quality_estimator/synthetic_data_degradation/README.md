# Synthetic Data Degradation for Training PACE's Quality & Reliability Estimation Models

The DroneVehicle dataset in its original form does not provide sufficient examples of localized poor sensor quality for training PACE's quality and reliability estimation models. This pipeline generates synthetic sensor degradations while preserving the synchronized RGB/IR image pairs and their object-detection annotations.

The goal is to produce, for every RGB/IR pair, an **8×8 quality map for each modality** indicating the estimated local quality of the sensor data corresponding to each spatial feature region.

---

## Dataset Generation Overview

Each original DroneVehicle sample consists of:

- an RGB image,
- an IR image,
- RGB XML ground-truth annotations, and
- IR XML ground-truth annotations.

The original DroneVehicle images contain a fixed **100-pixel white padding border on all four sides**. Before lighting analysis, synthetic degradation, or quality-label generation, this padding is removed.

For the standard DroneVehicle image dimensions:

```text
Original image:   840 × 712
Crop:             100 px from each side
Processed image:  640 × 512
```

The associated XML annotations are adjusted into the cropped coordinate system so that the ground-truth object locations remain aligned with the processed images.

Each processed example therefore contains:

- a cropped RGB image,
- a cropped IR image,
- adjusted RGB XML ground-truth annotations,
- adjusted IR XML ground-truth annotations,
- an 8×8 RGB quality map,
- an 8×8 IR quality map, and
- metadata describing any synthetic degradation applied.

Approximately **50% of training examples remain free of synthetic degradation**, while the remaining examples receive randomly sampled degradation.

All examples receive quality labels regardless of whether synthetic degradation is applied.

The original train/validation/test organization should ultimately be preserved to avoid leakage between related DroneVehicle frames.

---

## File Organization

```text
synthetic_data_degradation/
│
├── degradation.py
│   └── Generates irregular spatial masks and applies synthetic
│       degradations to images.
│
├── lighting.py
│   └── Analyzes RGB luminance.
│       Determines sufficiently illuminated regions and generates
│       the baseline 8×8 RGB quality map.
│
├── quality_labels.py
│   └── Converts pixel-level degradation information into 8×8
│       quality targets using degradation coverage, severity,
│       and baseline quality.
│
├── build_dataset.py
│   └── Orchestrates the complete dataset-generation pipeline.
│       Crops the fixed DroneVehicle padding, adjusts XML annotations,
│       computes baseline quality, samples and applies degradation,
│       generates quality labels, and saves the processed dataset.
│
├── visualize_quality.py
│   └── Validation/debugging utility for inspecting generated samples.
│       Displays:
│         1. the processed image,
│         2. the image with its 8×8 quality map overlaid, and
│         3. the image with adjusted XML ground-truth boxes overlaid.
│
└── README.md
    └── Documents the synthetic data-generation and validation procedure.
```

---

## Image Preprocessing and Padding Removal

DroneVehicle images contain a fixed 100-pixel padding region surrounding the actual sensor image.

The pipeline removes this padding before performing any quality analysis or degradation:

```python
cropped_image = image[
    100 : height - 100,
    100 : width - 100
]
```

For a standard `840 × 712` image, this produces:

```text
640 × 512
```

All subsequent operations are performed on this cropped image.

This is important because synthetic degradation and quality labels should describe **actual sensor information**, not the artificial white padding included in the stored dataset image.

After cropping, the 8×8 quality grid corresponds cleanly to:

```text
640 / 8 = 80 pixels per grid cell horizontally
512 / 8 = 64 pixels per grid cell vertically
```

Therefore, each quality-map cell corresponds to an `80 × 64` region of actual sensor imagery.

---

## XML Annotation Adjustment

Because the processed images are cropped, the original XML coordinates cannot simply be copied unchanged.

For each annotation coordinate:

```text
x_new = x_original - 100
y_new = y_original - 100
```

Objects entirely outside the retained sensor region are removed.

Objects intersecting the crop boundary are retained and their coordinates are clipped to the processed image dimensions:

```text
x ∈ [0, 639]
y ∈ [0, 511]
```

If the XML contains stored image dimensions, those dimensions are also updated to:

```text
width  = 640
height = 512
```

This keeps the processed image and its corresponding ground-truth annotations in the same coordinate system.

---

## Synthetic Degradations

Synthetic degradations are applied to irregular spatial regions rather than directly to the 8×8 feature grid.

This is important because real sensor degradations do not naturally align with model feature-map boundaries.

The currently supported degradation types are:

- **Water blur:** Gaussian blur representing water droplets, condensation, or dirty optics.
- **Motion blur:** directional blur representing camera or platform motion during capture.
- **Sensor noise:** Gaussian noise, line corruption, and salt-and-pepper noise representing electronic sensor artifacts.
- **Saturation:** localized overexposure or sensor saturation.

For a degraded example, the pipeline randomly samples:

- the modality to degrade,
- degradation type,
- degradation severity,
- one or more irregular spatial regions,
- degradation location, and
- degradation size.

The random seed is fixed and recorded to support reproducibility.

Current seed:

```text
42
```

---

## Baseline Quality

Quality labels represent **absolute local sensor quality**, rather than only whether synthetic degradation was injected.

### RGB

Original RGB imagery may already contain poorly illuminated regions. Therefore, clean RGB images should not automatically receive quality labels of `1.0` everywhere.

After cropping, the RGB image is converted into a luminance representation.

Local luminance is used to:

1. identify regions with sufficient visible information,
2. construct an 8×8 baseline RGB quality map, and
3. determine where synthetic RGB degradations can meaningfully be applied.

Dark regions receive lower baseline quality, while sufficiently illuminated regions approach a quality score of `1.0`.

A continuous luminance-to-quality mapping is used rather than a binary dark/light classification.

### IR

The original DroneVehicle IR imagery is treated as the reference-quality IR modality.

Therefore, after removing the padding, the baseline IR quality map is initialized as:

```text
8×8 matrix of 1.0 values
```

Synthetic degradation, if applied, subsequently reduces quality within affected regions.

---

## Lighting-Aware RGB Degradation

Synthetic RGB degradation is only applied where there is sufficient visible-light information to degrade.

For example, applying blur to an already nearly black RGB region may produce almost no observable change. Labeling that region as substantially degraded would create incorrect supervision for the quality estimator.

Therefore:

1. generate the random degradation mask,
2. compute the sufficiently-lit RGB mask,
3. intersect the two masks,
4. apply degradation only within the resulting effective mask.

Conceptually:

```text
effective degradation mask
        =
random degradation mask
        ×
sufficiently-lit RGB mask
```

Because padding has already been cropped out, the effective mask operates entirely on actual sensor imagery.

IR degradation does not use the RGB lighting constraint.

---

## 8×8 Quality-Map Generation

Synthetic degradation is applied at the full **cropped image resolution** and is **not aligned with the 8×8 grid**.

After degradation, an 8×8 grid is conceptually overlaid on the processed `640 × 512` image.

Each grid cell therefore represents:

```text
80 × 64 pixels
```

For every grid cell, the pipeline calculates:

```text
degradation coverage
    =
number of degraded pixels in the cell
    /
total number of pixels in the cell
```

This produces an 8×8 degradation-coverage map with values between `0` and `1`.

The final quality of patch `(i, j)` is currently computed using:

```text
Q_final(i,j)
    =
Q_base(i,j) × (1 - coverage(i,j) × severity)
```

Therefore:

- an untouched region retains its baseline quality,
- a partially degraded region experiences a proportional quality reduction,
- a fully degraded region receives the full severity penalty.

This formulation can later be refined if different degradation types require different severity-to-quality mappings.

---

## Clean vs. Degraded Examples

Every example receives quality labels.

### Clean RGB

```text
image:
    cropped original RGB

quality:
    luminance-derived 8×8 baseline map
```

### Degraded RGB

```text
image:
    cropped RGB with synthetic degradation

quality:
    baseline RGB quality adjusted by
    degradation coverage and severity
```

### Clean IR

```text
image:
    cropped original IR

quality:
    8×8 matrix of 1.0 values
```

### Degraded IR

```text
image:
    cropped IR with synthetic degradation

quality:
    baseline 1.0 quality adjusted by
    degradation coverage and severity
```

---

## Generated Dataset Structure

The generated dataset preserves each synchronized RGB/IR pair using a shared `pair_id`.

```text
processed_dataset/
│
├── rgb/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
│
├── ir/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
│
├── rgb_annotations/
│   ├── 00001.xml
│   ├── 00002.xml
│   └── ...
│
├── ir_annotations/
│   ├── 00001.xml
│   ├── 00002.xml
│   └── ...
│
├── quality_maps/
│   ├── 00001_rgb.npy
│   ├── 00001_ir.npy
│   ├── 00002_rgb.npy
│   ├── 00002_ir.npy
│   └── ...
│
└── metadata.csv
```

For example, pair `00001` consists of:

```text
RGB image
    → rgb/00001.jpg

RGB ground truth
    → rgb_annotations/00001.xml

IR image
    → ir/00001.jpg

IR ground truth
    → ir_annotations/00001.xml

RGB quality
    → quality_maps/00001_rgb.npy

IR quality
    → quality_maps/00001_ir.npy

metadata
    → row 00001 in metadata.csv
```

The RGB image, IR image, XML annotations, and quality maps all describe the same processed spatial sample.

---

## Metadata

`metadata.csv` serves as the master index for the generated dataset.

Each row represents one complete synchronized RGB/IR example.

| Field              | Purpose                                                 |
| ------------------ | ------------------------------------------------------- |
| `pair_id`          | Unique identifier shared by all files in the pair       |
| `split`            | Original train/validation/test split                    |
| `rgb_path`         | Path to processed/cropped RGB image                     |
| `ir_path`          | Path to processed/cropped IR image                      |
| `rgb_xml`          | Path to adjusted RGB ground-truth XML                   |
| `ir_xml`           | Path to adjusted IR ground-truth XML                    |
| `rgb_quality`      | Path to RGB 8×8 quality map                             |
| `ir_quality`       | Path to IR 8×8 quality map                              |
| `degraded`         | Whether synthetic degradation was applied               |
| `modality`         | Modality receiving degradation (`rgb`, `ir`, or `none`) |
| `degradation_type` | Synthetic degradation type                              |
| `severity`         | Sampled degradation severity                            |
| `source_rgb`       | Path to original RGB image                              |
| `source_ir`        | Path to original IR image                               |
| `seed`             | Random seed used during generation                      |

The actual 8×8 quality maps are stored as `.npy` files rather than embedded directly into the CSV.

---

## Dataset Generation

The current build script can initially be run on a small number of samples for validation.

From the PACE repository root:

```bash
python prototyping/quality_estimator/synthetic_data_degradation/build_dataset.py
```

During development:

```python
MAX_SAMPLES = 10
```

can be used to restrict generation to a small test set.

Once the pipeline has been fully validated, this can be changed to:

```python
MAX_SAMPLES = None
```

for full dataset generation.

If regenerating the dataset after changing the generation logic, remove the old generated output first:

```bash
rm -rf processed_dataset
```

and rerun the build script.

---

## Visualization and Dataset Validation

`visualize_quality.py` is provided as a debugging and validation tool for inspecting individual generated examples before scaling dataset generation to the full dataset.

Run it from the PACE repository root using:

```bash
python prototyping/quality_estimator/synthetic_data_degradation/visualize_quality.py <pair_id> <modality>
```

For example:

```bash
python prototyping/quality_estimator/synthetic_data_degradation/visualize_quality.py 00004 ir
```

or:

```bash
python prototyping/quality_estimator/synthetic_data_degradation/visualize_quality.py 00004 rgb
```

The script loads the corresponding files from `processed_dataset/` and displays **three images side-by-side**:

```text
┌────────────────────┐
│  Processed Image   │
└────────────────────┘

          +

┌────────────────────┐
│ Processed Image +  │
│ 8×8 Quality Grid   │
└────────────────────┘

          +

┌────────────────────┐
│ Processed Image +  │
│ XML Bounding Boxes │
└────────────────────┘
```

More specifically:

### Panel 1 — Processed Image

Displays the final cropped image exactly as it will appear in the generated dataset.

This panel should be used to verify:

- the 100-pixel padding has been removed,
- the image dimensions are correct,
- synthetic degradation appears spatially reasonable, and
- no unintended geometric transformation has occurred.

For standard DroneVehicle images, the displayed image should have dimensions:

```text
512 × 640 × 3
```

in NumPy `(height, width, channels)` notation.

### Panel 2 — Quality Map Overlay

Displays the same image with the 8×8 quality grid overlaid.

Each cell contains its corresponding quality value.

This panel should be used to verify that:

- the grid spans only the cropped sensor image,
- the grid is spatially aligned with the image,
- each grid cell corresponds to the expected `80 × 64` pixel region,
- degraded regions receive reduced quality values,
- unaffected regions retain their baseline quality,
- RGB dark regions receive appropriately reduced baseline quality.

The script also prints the quality-map shape and value range.

The expected quality-map shape is:

```text
(8, 8)
```

### Panel 3 — XML Ground-Truth Overlay

Displays the processed image with object boxes derived from the **adjusted processed XML annotation**.

The visualization reads from:

```text
processed_dataset/rgb_annotations/
```

for RGB or:

```text
processed_dataset/ir_annotations/
```

for IR.

This panel is used to verify that the 100-pixel coordinate adjustment performed during cropping correctly preserves alignment between the objects in the image and their ground-truth annotations.

The displayed boxes should surround the same physical objects they annotated in the original DroneVehicle image.

---

## Recommended Validation Before Full Generation

Before generating the complete training dataset, inspect multiple examples with `visualize_quality.py`.

At minimum, validation should include:

1. a clean RGB example,
2. a degraded RGB example,
3. a clean IR example,
4. a degraded IR example,
5. examples with objects near image boundaries,
6. examples containing low-light RGB regions.

For each example, verify:

```text
Image:
    padding removed correctly
    dimensions = 640 × 512

Quality map:
    shape = 8 × 8
    grid aligned with image
    quality reductions correspond to degraded regions
    RGB baseline reflects lighting

Annotations:
    boxes remain aligned after crop
    no 100-pixel coordinate offset remains
```

This validation should be completed before setting:

```python
MAX_SAMPLES = None
```

and generating the full dataset.

---

## Complete Generation Flow

For each synchronized RGB/IR pair:

1. Load the RGB image, IR image, and both XML annotations.
2. Record the original image dimensions.
3. Remove the fixed 100-pixel border from RGB and IR.
4. Transform the RGB and IR XML annotations into the cropped coordinate system.
5. Compute RGB luminance information on the cropped RGB image.
6. Generate the baseline 8×8 RGB quality map.
7. Initialize the baseline 8×8 IR quality map to `1.0`.
8. Randomly determine whether the example receives synthetic degradation.
9. If clean:
   - preserve the cropped RGB and IR images,
   - preserve their baseline quality maps.
10. If degraded:
    - randomly select the modality to degrade,
    - randomly select degradation type and severity,
    - generate an irregular spatial degradation mask,
    - for RGB, restrict the mask to sufficiently illuminated regions,
    - apply the degradation,
    - overlay the 8×8 grid conceptually,
    - calculate degradation coverage for every patch,
    - combine baseline quality, coverage, and severity to produce the final quality map.
11. Save the processed RGB and IR images.
12. Save the adjusted RGB and IR XML annotations.
13. Save both 8×8 quality maps.
14. Add one row to `metadata.csv` describing the complete pair and any degradation applied.
15. Use `visualize_quality.py` on sample outputs to validate image, quality-map, and annotation alignment.

The resulting dataset therefore maintains the structure:

```text
cropped RGB + adjusted RGB ground truth
        +
cropped IR + adjusted IR ground truth
        +
RGB 8×8 quality map
        +
IR 8×8 quality map
        +
pair-level degradation metadata
```

This provides the synchronized multimodal inputs, detection ground truth, and local sensor-quality supervision required for training and evaluating PACE's downstream quality and reliability estimation components.
