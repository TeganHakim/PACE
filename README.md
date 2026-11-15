# PACE
Patch-Adaptive Cross-Attention mechanism for Efficient sensor fusion

## Setup

Clone the repository:

```bash
git clone https://github.com/TeganHakim/PACE.git
cd PACE
```

Create and activate the conda environment, then install the dependencies:

```bash
conda create -n pace python=3.7
conda activate pace
pip install -r requirements.txt
```

## Dataset

For this research project, we engineered a custom dataset based on the [VisDrone-DroneVehicle](https://www.kaggle.com/datasets/brendanalvey/visdrone-dronevehicle/data?select=VisDrone-DroneVehicle) dataset from Kaggle. For details on how to generate the dataset, see [Dataset Generation Instructions for Reproducibility](prototyping/quality_estimator/synthetic_data_degradation/README.md#dataset-generation-instructions-for-reproducibility).

### Realistic Degraded Test Dataset

To evaluate the robustness of PACE under localized sensor degradation, we generated an additional test dataset using the original VisDrone-DroneVehicle RGB/IR test images.

Each image pair contains one randomly selected degraded modality (RGB or IR), while the other modality remains free of synthetic degradation. Two types of degradation are applied:

- **Sensor noise:** Severe noise introduced across 3–6 horizontal or vertical bands.
- **Localized blur:** Strong Gaussian blur applied to 3–6 irregular, droplet-shaped regions.

The degradation regions cover approximately 50% of the image and preferentially overlap annotated vehicles, allowing us to evaluate fusion performance when important visual information is corrupted.

All images are cropped to 640 × 512 pixels using the same preprocessing procedure as the original PACE dataset. The dataset also includes object annotations, binary degradation masks, and metadata describing the applied degradation.

This dataset is intended for evaluating the effectiveness of quality-aware selective fusion compared with unconditional fusion baselines, including 50/50 image addition.

**Download:** [Realistic Degraded Test Processed Dataset (Google Drive)](https://drive.google.com/file/d/1o3a4KJ8E6elQBhFz1uGteNtOOEI43QmX/view?usp=drive_link)

After downloading, extract the archive:

```bash
unzip realistic_degraded_test_processed_dataset.zip
```

The extracted dataset has the following structure:

```text
processed_dataset_realistic_degraded_v2/
├── test/
│   ├── rgb/
│   ├── ir/
│   ├── rgb_annotations/
│   ├── ir_annotations/
│   └── degradation_masks/
├── metadata.csv
└── realistic_degradation_config.json
```

## Train Autoencoders

This was done on a NVIDIA Jetson AGX Thor Developer Kit.

#### Train the RGB + IR Autoencoder

Run the following commands each with different `type_flag` values in `baseline/CrossFuse/args_auto.py` to train the RGB and IR autoencoders separately. Make sure to do so inside a tmux session to avoid interruptions during training.

```bash
cd baseline/CrossFuse
python3 train_autoencoder.py
```
