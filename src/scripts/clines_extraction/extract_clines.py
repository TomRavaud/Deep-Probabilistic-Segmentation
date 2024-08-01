"""
Script to compute and save correspondence lines coordinates for each object in a
dataset (MegaPose format). Lines can then be extracted from the images during the
data loading process.
"""
# Standard libraries
from pathlib import Path
import json
import sys
import multiprocessing
from functools import partial
from time import time

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import libarchive

# Custom modules
from toolbox.geometry.clines import (
    extract_contour_points_and_normals,
    extract_contour_lines,
    extract_only_largest_contour,
    random_homography_from_points,
    apply_homography_to_points_with_normals,
)


def extract_clines(shard_ids: list[str], config: dict) -> None:
    """Extract correspondence lines for each object in some shards of a dataset.

    Args:
        shard_ids (list[str]): List of shard ids to process.
        config (dict): Configuration dictionary with the following keys:
            - num_points_on_contour (int): Number of points to extract from the contour.
            - line_size_half (int): Half the size of the lines to extract.
            - homography_scale (float): Scale of the homography to apply.
            - min_area (int): Minimum area of the object to consider it.
            - data_path (str): Path to the dataset.
            - dataset_name (str): Name of the dataset.
    """
    for shard_id in shard_ids:

        print(f"Processing shard {shard_id}")

        # Load the shard
        shard_path =\
            Path(config["data_path"]) / config["dataset_name"] / f"{shard_id}.tar"
        # Create the output directory if it does not exist
        output_path =\
            Path(config["data_path"]) /\
                (config["dataset_name"]+"_clines_to_remove") / shard_id
        output_path.mkdir(exist_ok=True, parents=True)

        start = time()
        
        # Process the images
        with libarchive.Archive(shard_path.as_posix()) as shard:
            
            counter = 0
            
            for entry in shard:
                if "segmentation.png" in entry.pathname:
                    # Read the segmentation image
                    segmentation_file = shard.read(size=entry.size)
                    counter += 1
                elif "object_datas.json" in entry.pathname:
                    # Read the object data
                    obj_data_file = shard.read(size=entry.size)
                    counter += 1
                
                if counter == 2:
                    # Reset the counter
                    counter = 0
                    
                    # Load the semantic segmentation data
                    segmentation = cv2.imdecode(
                        np.frombuffer(segmentation_file, np.uint8),
                        cv2.IMREAD_UNCHANGED,
                    )
                    # Load the object data
                    obj_data = json.loads(obj_data_file)
                    
            
                    # Get the unique visible ids in the segmentation
                    unique_ids_visible = set(np.unique(segmentation))
                    
                    img_id = entry.pathname.split(".")[0]
                    
                    # Create a directory for the image
                    (output_path / img_id).mkdir(exist_ok=True)
                    
                    # List of object ids
                    oids = []
                    # List of correspondence lines to save
                    clines_list = []
                    
                    # Process each object
                    for obj in obj_data:
                        
                        try:
                            # Get the object id
                            oid = obj["unique_id"]

                            # If the files already exist, skip
                            if (output_path / f"{img_id}_{oid}.clines.rgb.npy").exists()\
                                and (output_path /\
                                    f"{img_id}_{oid}.clines.seg.npy").exists():
                                    continue

                            # Get the bounding box of the visible part of the object
                            bbox_modal = np.array(obj["bbox_modal"])
                            # Area of the box
                            area = (bbox_modal[3] - bbox_modal[1])\
                                * (bbox_modal[2] - bbox_modal[0])

                            # Filter objects with low visibility
                            if oid not in unique_ids_visible or\
                                np.any(bbox_modal < 0) or area < config["min_area"]:
                                    continue
                            
                            # Extract the binary mask for the object
                            mask = (segmentation == oid).astype(np.uint8)
                            mask = extract_only_largest_contour(mask)
                            points, normals = extract_contour_points_and_normals(
                                mask,
                                num_points_on_contour=config["num_points_on_contour"],
                            )
                            if points is None:
                                continue
                            H = random_homography_from_points(
                                points,
                                scale=config["homography_scale"],
                            )
                            points_transformed, normals_transformed = (
                                apply_homography_to_points_with_normals(
                                    points,
                                    normals, H,
                                )
                            )
                            clines = extract_contour_lines(
                                points_transformed,
                                normals_transformed,
                                line_size_half=config["line_size_half"],
                            )
                            if clines is None:
                                continue
                            
                            clines = clines.astype(np.int32)
                            
                        except Exception as e:
                            print(f"Error in {img_id}_{oid}: {e}")
                            continue

                        oids.append(oid)
                        clines_list.append(clines)
                        
                    # Save the contour lines
                    for oid, clines in zip(oids, clines_list):
                        np.save(output_path / img_id / f"{oid}.clines.npy", clines)
        
        print(f"Shard {shard_id} processed in {time()-start:.2f} seconds")


if __name__ == "__main__":
    #------------------------------------------#
    # Parameters #
    #------------------------------------------#
    config = {
        "num_points_on_contour": 200,
        "line_size_half": 60,
        "homography_scale": 0.1,
        "min_area": 9000,
        "data_path": "data/webdatasets",
        "dataset_name": "gso_1M",
    }
    #------------------------------------------#
    
    # Create partial function
    extract_clines_partial = partial(extract_clines, config=config)
    
    # Create the output directory if it does not exist
    (Path(config["data_path"]) /\
        (config["dataset_name"]+"_clines_to_remove")).mkdir(exist_ok=True)

    # Get the shard ids
    dataset_path = Path(config["data_path"]) / config["dataset_name"]
    shard_ids = sorted(
        shard.stem for shard in dataset_path.iterdir() if shard.suffix == ".tar"
    )
    
    shard_ids_split = np.array_split(shard_ids, 20)
    
    with multiprocessing.Pool(20) as pool:
        pool.map(extract_clines_partial, shard_ids_split)
