# Deep Probabilistic Segmentation

## Description

Development of a deep probabilistic binary segmentation model for the segmentation of an object to be tracked in a sequence of monocular RGB images. The aim is not to produce a mask for the current image, as is customary, but a segmentation based on color statistics that can be used on several consecutive images in the sequence (assuming the statistics are almost constant).


## Installation

You first need to clone the repository main branch:

```bash
git clone https://github.com/TomRavaud/deep_probabilistic_segmentation.git
cd deep_probabilistic_segmentation
```

Then, you need to install the required dependencies. The project uses conda to manage the dependencies. You can create a new conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

## Dataset download

The dataset used to train the model is the Google Scanned Objects split of the [MegaPose](https://github.com/megapose6d/megapose6d) dataset. You can download the dataset, as well as the 3D models, by running the script `data/download_gso_data.sh` in the directory in which you want to store the dataset. For example, to download the dataset in the `data` directory of the repository, run:

```bash
cd data
bash download_gso_data.sh
```
To avoid downloading the whole dataset, this script allows you to specify the number of shards to download. Follow the displayed instructions when running it.

## Usage

To activate the conda environment, run:

```bash
conda activate deep_probabilistic_segmentation
```

You can now run scripts located in the `src/scripts` directory from the root of the repository. For example, to train the model, run:

```bash
python src/scripts/train.py
```


## Acknowledgement

Some of the code is borrowed from [MegaPose](https://github.com/megapose6d/megapose6d) (maintained in [happypose](https://github.com/agimus-project/happypose/tree/dev)) so as to make the dataset handling easier.
