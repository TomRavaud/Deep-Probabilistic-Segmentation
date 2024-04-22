# Deep Probabilistic Segmentation

## Description

Development of a deep probabilistic binary segmentation model for the segmentation of an object to be tracked in a sequence of monocular RGB images. The aim is not to produce a mask for the current image, as is customary, but a segmentation based on color and texture statistics that can be used on several consecutive images in the sequence (assuming the statistics are constant).


## Installation

You first need to clone the repository main branch:

```bash
git clone https://github.com/TomRavaud/deep_probabilistic_segmentation.git
cd deep_probabilistic_segmentation
```

Then, you need to install the required dependencies. The project uses conda to manage the dependencies. You can create a new conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yml
```

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

Some of the code is borrowed from [MegaPose](https://github.com/megapose6d/megapose6d), maintained in [happypose](https://github.com/agimus-project/happypose/tree/dev). It made it easier to get to grips with the dataset used to train MegaPose.
