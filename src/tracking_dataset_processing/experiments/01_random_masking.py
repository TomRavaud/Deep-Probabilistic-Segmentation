#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-07-17
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

# Standard libraries
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


# Parameters
output_path = Path("to_remove")
sample_names = [o.name.split(".")[0] for o in output_path.glob("*.clines.rgb.npy")]


for sample_name in sample_names:
    
    rgb = np.load(output_path / f"{sample_name}.clines.rgb.npy")
    seg = np.load(output_path / f"{sample_name}.clines.seg.npy")
    mask = np.isnan(seg)

    clip_width = np.random.randint(rgb.shape[1] / 2 - 1)
    mask[:, :clip_width] = True
    mask[:, -clip_width:] = True

    # np.sum(np.diff(seg[:, clip_width:-clip_width][0])) == -1
    # and np.sum(np.abs(np.diff(seg[:, clip_width:-clip_width][0]))) == 1
    # mask lines where there is more than one change or the change is not to -1
    m = np.bitwise_and(
        np.isclose(
            np.sum(np.diff(seg[:, clip_width:-clip_width], axis=-1), axis=-1), -1
        ),
        np.isclose(
            np.sum(np.abs(np.diff(seg[:, clip_width:-clip_width], axis=-1)), axis=-1), 1
        ),
    )
    mask[:, clip_width:-clip_width] = ~np.repeat(
        m[:, np.newaxis], seg.shape[1] - clip_width * 2, axis=1
    )

    seg_with_boundary_mask = seg.copy()
    seg_with_boundary_mask[mask] = np.nan

    masked_rgb = rgb.copy().astype(np.uint8)
    masked_rgb[mask] = 0

    masked_seg = 255 * seg.copy().astype(np.uint8)
    masked_seg[mask] = 127

    fig: plt.Figure
    fig, axes = plt.subplots(1, 2, squeeze=False, sharex=True, sharey=True)
    ax: plt.Axes = axes[0, 0]
    ax.imshow(masked_rgb)
    ax.axis("off")
    ax.set_title("RGB")
    ax: plt.Axes = axes[0, 1]
    ax.imshow(masked_seg, cmap="bwr")
    ax.axis("off")
    ax.set_title("Segmentation")
    plt.show()

    # fig.savefig('tmp2.png')

