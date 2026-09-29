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

Download the [VisDrone-DroneVehicle](https://www.kaggle.com/datasets/brendanalvey/visdrone-dronevehicle/data?select=VisDrone-DroneVehicle) dataset from Kaggle and unzip it into the top level of the repository. The folder layout should look like this:

```
PACE/
├── baseline/
│   └── CrossFuse/
├── pace/
├── prototyping/
├── reports/
├── requirements.txt
└── VisDrone-DroneVehicle/
    ├── train/
    │   ├── trainimg/       # visible (RGB) images
    │   ├── trainimgr/      # infrared (IR) images
    │   ├── trainlabel/     # visible XML annotations
    │   └── trainlabelr/    # infrared XML annotations
    ├── val/
    │   ├── valimg/
    │   ├── valimgr/
    │   ├── vallabel/
    │   └── vallabelr/
    └── test/
        ├── testimg/
        ├── testimgr/
        ├── testlabel/
        └── testlabelr/
```

Files share the same name across folders, so `00001.jpg` in `trainimg/` pairs with `00001.jpg` in `trainimgr/`. The `r` suffix marks the infrared folders.

## Corrupted images

The Kaggle page for the dataset notes:

> "There are a few train images that appear to be corrupted / missing. They failed to unzip from the source I got this dataset from."

There are 28 of them. They will crash training partway since `The type of image (...) is None`, so remove them before training. Every sample needs all four parts (RGB image, IR image, RGB label, IR label), so if one image is bad, the whole sample is deleted.

From `baseline/CrossFuse`, first scan the dataset:

```bash
python dataset_tools.py check
```

This lists every unreadable image and writes their names to `VisDrone-DroneVehicle/bad_images.txt`. You should see 28.

Then delete them:

```bash
python dataset_tools.py delete
```

This lists every file it is about to remove and asks you to type `yes`. Pass `--yes` to skip the prompt. Afterwards `bad_images.txt` is renamed to `bad_images.txt.done`.

Both commands assume the dataset is at `../VisDrone-DroneVehicle`. To use a different location, pass the path after the command, e.g. `python dataset_tools.py check /path/to/VisDrone-DroneVehicle`.