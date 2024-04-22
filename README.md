# Deep Probabilistic Segmentation

## Description

Development of a deep probabilistic binary segmentation model for the segmentation of an object to be tracked in a sequence of monocular RGB images. The aim is not to produce a mask for the current image, as is customary, but a segmentation based on color and texture statistics that can be used on several consecutive images in the sequence (assuming the statistics are constant).


## Installation

### Conda environment

To create a conda environment with all the necessary dependencies, run the following command:

```bash
conda env create -f environment.yml
```
