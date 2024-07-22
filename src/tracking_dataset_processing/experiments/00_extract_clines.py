# Standard libraries
from pathlib import Path
import json
import sys
import tarfile
import multiprocessing

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom modules
from tracking_dataset_processing.tracking_dataset.clines import (
    extract_contour_points_and_normals,
    extract_contour_lines,
    extract_only_largest_contour,
    random_homography_from_points,
    apply_homography_to_points_with_normals,
)

#------------#
# Parameters #
#------------#
NUM_POINTS_ON_CONTOUR = 200
LINE_SIZE_HALF = 60
HOMOGRAPHY_SCALE = 0.1
# Minimum area for an object to be considered (bounding box area of the visible part)
MIN_AREA = 9000
SAVE_FIGURES = False
DATA_PATH = Path("data/webdatasets")
DATASET_NAME = "gso_1M"
#------------#


def extract_clines(chunk_ids):
    
    for chunk_id in chunk_ids:

        print(f"Processing chunk {chunk_id}")

        # Load the chunk
        chunk_path = DATA_PATH / DATASET_NAME / f"{chunk_id}.tar"
        chunk_tar = tarfile.open(chunk_path, "r")
        # Create the output directory if it does not exist
        output_path = DATA_PATH / f"{DATASET_NAME}_clines" / chunk_id
        output_path.mkdir(exist_ok=True)

        # Extract unique sample ids
        image_ids = list(
            set(path.name.split(".")[0] for path in chunk_tar.getmembers())
        )

        # Process the images
        for img_id in image_ids:
        # for img_id in tqdm(image_ids):

            # Extract the RGB image
            rgb_file = chunk_tar.extractfile(f"{img_id}.rgb.png")
            # Load the RGB image
            rgb = cv2.imdecode(
                np.frombuffer(rgb_file.read(), np.uint8), cv2.IMREAD_COLOR
            )

            # Extract the semantic segmentation data
            segmentation_file = chunk_tar.extractfile(f"{img_id}.segmentation.png")
            # Load the semantic segmentation data
            segmentation = cv2.imdecode(
                np.frombuffer(segmentation_file.read(), np.uint8),
                cv2.IMREAD_UNCHANGED,
            )

            # Extract the object data
            obj_data_file = chunk_tar.extractfile(f"{img_id}.object_datas.json")
            # Load the object data
            obj_data = json.loads(obj_data_file.read())

            # Get the unique visible ids in the segmentation
            unique_ids_visible = set(np.unique(segmentation))

            # Process each object
            for obj in obj_data:

                # Get the bounding box of the visible part of the object
                bbox_modal = np.array(obj["bbox_modal"])
                # Area of the box
                area = (bbox_modal[3] - bbox_modal[1]) * (bbox_modal[2] - bbox_modal[0])

                # Filter objects with low visibility
                if obj["unique_id"] not in unique_ids_visible or\
                    np.any(bbox_modal < 0) or area < MIN_AREA:
                        continue

                # Get the object id
                oid = obj["unique_id"]
                # Extract the binary mask for the object
                mask = (segmentation == oid).astype(np.uint8)

                mask = extract_only_largest_contour(mask)
                points, normals = extract_contour_points_and_normals(
                    mask, num_points_on_contour=NUM_POINTS_ON_CONTOUR
                )
                H = random_homography_from_points(points, scale=HOMOGRAPHY_SCALE)
                points_transformed, normals_transformed = (
                    apply_homography_to_points_with_normals(points, normals, H)
                )

                clines = extract_contour_lines(
                    points_transformed,
                    normals_transformed,
                    line_size_half=LINE_SIZE_HALF
                )

                # Find points that are inside and outside the image
                clines_valid = np.bitwise_and(
                    np.all(clines >= (0, 0), axis=-1),
                    np.all(clines < np.array(mask.shape), axis=-1),
                )
                # Fill the contour lines with RGB data, and NaN for points outside the
                # image
                clines_rgb = np.ones(clines.shape[:2] + (3,)) * np.nan
                clines_rgb[clines_valid] = rgb[
                    clines[clines_valid][:, 0], clines[clines_valid][:, 1]
                ]
                # Set the segmentation data for the contour lines, and NaN for points
                # outside the image
                clines_seg = np.ones(clines.shape[:2]) * np.nan
                clines_seg[clines_valid] = mask[
                    clines[clines_valid][:, 0], clines[clines_valid][:, 1]
                ]

                # Save the contour lines
                np.save(output_path / f"{img_id}_{oid}.clines.rgb.npy", clines_rgb)
                np.save(output_path / f"{img_id}_{oid}.clines.seg.npy", clines_seg)


                # Save the figures if required
                if SAVE_FIGURES:
                    rgb_with_clines = rgb.copy()
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
                    plt.close(fig)


if __name__ == "__main__":

    # Create the output directory if it does not exist
    (DATA_PATH / f"{DATASET_NAME}_clines").mkdir(exist_ok=True)

    # Get the chunk ids
    dataset_path = DATA_PATH / DATASET_NAME
    chunk_ids = sorted(
        chunk.stem for chunk in dataset_path.iterdir() if chunk.suffix == ".tar"
    ) 
    
    # Split the chunk ids into chunks for multiprocessing
    chunk_ids_split = np.array_split(chunk_ids, multiprocessing.cpu_count())
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(extract_clines, chunk_ids_split)
