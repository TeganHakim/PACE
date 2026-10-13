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

## Train Autoencoders
This was done on a NVIDIA Jetson AGX Thor Developer Kit.

#### Train the RGB + IR Autoencoder
Run the following commands each with different `type_flag` values in `baseline/CrossFuse/args_auto.py` to train the RGB and IR autoencoders separately. Make sure to do so inside a tmux session to avoid interruptions during training.
```bash
cd baseline/CrossFuse
python3 train_autoencoder.py
```