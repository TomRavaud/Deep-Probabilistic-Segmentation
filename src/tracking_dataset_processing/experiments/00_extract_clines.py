#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-07-16
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

# Standard libraries
from pathlib import Path
import json
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from tracking_dataset_processing.tracking_dataset.clines import (
    extract_contour_points_and_normals,
    extract_contour_lines,
    extract_only_largest_contour,
    random_homography_from_points,
    apply_homography_to_points_with_normals,
)


# Parameters
num_points_on_contour = 200
line_size_half = 60

homography_scale = 0.05

# data_path = Path("data/webdatasets/00000000")
# chunks_paths = [p for p in data_path.iterdir()]
chunk_path = Path("data/webdatasets/00000000")
output_path = Path("to_remove")


# Extract sample ids
image_ids = [path.name.split(".")[0] for path in chunk_path.glob("*.rgb.png")]

output_path.mkdir(exist_ok=True)

# Process the first three images
for img_id in image_ids[:3]:

    # Load the RGB image
    rgb = cv2.imread(str(chunk_path / f"{img_id}.rgb.png"))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Load the semantic segmentation data
    segmentation = cv2.imread(
        str(chunk_path / f"{img_id}.segmentation.png"), cv2.IMREAD_UNCHANGED
    )
    
    # Load the object data
    obj_data = json.load(open(str(chunk_path / f"{img_id}.object_datas.json")))

    
    for obj in obj_data:
        
        # Filter objects with low visibility
        if obj["visib_fract"] < 0.5:
            continue
        
        # Get the object id
        oid = obj["unique_id"]
        # Extract the binary mask for the object
        mask = (segmentation == oid).astype(np.uint8)

        mask = extract_only_largest_contour(mask)
        points, normals = extract_contour_points_and_normals(
            mask, num_points_on_contour=num_points_on_contour
        )
        H = random_homography_from_points(points, scale=homography_scale)
        points_transformed, normals_transformed = (
            apply_homography_to_points_with_normals(points, normals, H)
        )

        clines = extract_contour_lines(
            points_transformed, normals_transformed, line_size_half=line_size_half
        )

        clines_valid = np.bitwise_and(
            np.all(clines >= (0, 0), axis=-1),
            np.all(clines < np.array(mask.shape), axis=-1),
        )
        clines_rgb = np.ones(clines.shape[:2] + (3,)) * np.nan
        clines_rgb[clines_valid] = rgb[
            clines[clines_valid][:, 0], clines[clines_valid][:, 1]
        ]

        clines_seg = np.ones(clines.shape[:2]) * np.nan
        clines_seg[clines_valid] = mask[
            clines[clines_valid][:, 0], clines[clines_valid][:, 1]
        ]
        
        print(clines_rgb.shape)
        
        # imageio.imwrite(output_path / f"{img_id}_{oid}.clines.rgb.png", clines_rgb.astype(np.uint8))
        # imageio.imwrite(output_path / f"{img_id}_{oid}.clines.seg.png", 255 * clines_seg.astype(np.uint8))
        np.save(output_path / f"{img_id}_{oid}.clines.rgb.npy", clines_rgb)
        np.save(output_path / f"{img_id}_{oid}.clines.seg.npy", clines_seg)

        rgb_with_clines = rgb.copy()
        # rgb_with_clines[*clines[clines_valid].T] = (0, 0, 255)
        clines_x, clines_y = clines[clines_valid].T
        rgb_with_clines[clines_x, clines_y] = (0, 0, 255)
        
        fig: plt.Figure
        fig, axes = plt.subplots(2, 2, squeeze=False)
        ax: plt.Axes = axes[0, 0]
        ax.imshow(rgb_with_clines)
        ax.set_title("RGB")
        ax.axis("off")
        ax: plt.Axes = axes[0, 1]
        ax.imshow(mask.astype(np.uint8) * 255, cmap="bwr")
        ax.set_title("Segmentation")
        ax.axis("off")

        ax: plt.Axes = axes[1, 0]
        rgb_show = clines_rgb.copy()
        rgb_show[np.isnan(rgb_show)] = 0
        ax.imshow(rgb_show.astype(np.uint8))
        ax.set_title("Contour lines RGB")
        ax.axis("off")

        ax: plt.Axes = axes[1, 1]
        seg_show = clines_seg.copy()
        seg_show *= 255
        seg_show[np.isnan(seg_show)] = 128
        ax.imshow(seg_show.astype(np.uint8), cmap="bwr", vmin=0, vmax=255)
        ax.set_title("Contour lines Segmentation")
        ax.axis("off")

        fig.savefig(output_path / f"clines_{img_id}_{oid}.png")
        plt.show()
        plt.close(fig)
        # break