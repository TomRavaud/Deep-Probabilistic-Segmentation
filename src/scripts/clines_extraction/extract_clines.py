# Standard libraries
from pathlib import Path
import json
import sys
# import tarfile
import libarchive
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

# Custom modules
from toolbox.geometry.clines import (
    extract_contour_points_and_normals,
    extract_contour_lines,
    extract_only_largest_contour,
    random_homography_from_points,
    apply_homography_to_points_with_normals,
)


def extract_clines(shard_ids, config):
    
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
                
                # if "rgb.png" in entry.pathname:
                #     # Read the RGB image
                #     rgb_file = shard.read(size=entry.size)
                #     counter += 1
                if "segmentation.png" in entry.pathname:
                # elif "segmentation.png" in entry.pathname:
                    # Read the segmentation image
                    segmentation_file = shard.read(size=entry.size)
                    counter += 1
                elif "object_datas.json" in entry.pathname:
                    # Read the object data
                    obj_data_file = shard.read(size=entry.size)
                    counter += 1
                
                if counter == 2:
                # if counter == 3:
                    
                    # Reset the counter
                    counter = 0
                    
                    # # Load the RGB image
                    # rgb = cv2.imdecode(
                    #     np.frombuffer(rgb_file, np.uint8), cv2.IMREAD_COLOR
                    # )
                    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
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
                    
                    oids = []
                    # clines_rgb_list = []
                    # clines_seg_list = []
                    clines_list = []
                    
                    # Process each object
                    for obj in obj_data:
                        
                        try:
                            # Get the object id
                            oid = obj["unique_id"]

                            # # If the files already exist, skip
                            # if (output_path / f"{img_id}_{oid}.clines.rgb.npy").exists() and\
                            #     (output_path / f"{img_id}_{oid}.clines.seg.npy").exists():
                            #         continue

                            # Get the bounding box of the visible part of the object
                            bbox_modal = np.array(obj["bbox_modal"])
                            # Area of the box
                            area = (bbox_modal[3] - bbox_modal[1]) * (bbox_modal[2] - bbox_modal[0])

                            # Filter objects with low visibility
                            if oid not in unique_ids_visible or\
                                np.any(bbox_modal < 0) or area < config["min_area"]:
                                    continue
                            
                            # Extract the binary mask for the object
                            mask = (segmentation == oid).astype(np.uint8)
                            mask = extract_only_largest_contour(mask)
                            points, normals = extract_contour_points_and_normals(
                                mask, num_points_on_contour=config["num_points_on_contour"]
                            )
                            if points is None:
                                continue
                            H = random_homography_from_points(points, scale=config["homography_scale"])
                            points_transformed, normals_transformed = (
                                apply_homography_to_points_with_normals(points, normals, H)
                            )
                            clines = extract_contour_lines(
                                points_transformed,
                                normals_transformed,
                                line_size_half=config["line_size_half"],
                            )
                            if clines is None:
                                continue
                            
                            clines = clines.astype(np.int32)
                            
                            # # Find points that are inside and outside the image
                            # clines_valid = np.bitwise_and(
                            #     np.all(clines >= (0, 0), axis=-1),
                            #     np.all(clines < np.array(mask.shape), axis=-1),
                            # )
                            
                            # # Fill the contour lines with RGB data, and 0 for points outside the
                            # # image
                            # clines_rgb = np.zeros(clines.shape[:2] + (3,), np.uint8)
                            # clines_rgb[clines_valid] = rgb[
                            #     clines[clines_valid][:, 0], clines[clines_valid][:, 1]
                            # ]
                            # # Set the segmentation data for the contour lines, and 5 for points
                            # # outside the image
                            # clines_seg = np.ones(clines.shape[:2], np.uint8) * 5
                            # clines_seg[clines_valid] = mask[
                            #     clines[clines_valid][:, 0], clines[clines_valid][:, 1]
                            # ]
                        
                        except Exception as e:
                            print(f"Error in {img_id}_{oid}: {e}")
                            continue

                        # Save the contour lines
                        # np.save(output_path / f"{img_id}_{oid}.clines.rgb.npy", clines_rgb)
                        # np.save(output_path / f"{img_id}_{oid}.clines.seg.npy", clines_seg)
                        
                        oids.append(oid)
                        # clines_rgb_list.append(clines_rgb)
                        # clines_seg_list.append(clines_seg)
                        clines_list.append(clines)
                        
                        
                        
                        # Save the figures if required
                        if config["save_figures"]:
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
                    
                    # Save the contour lines
                    for oid, clines in zip(oids, clines_list):
                        np.save(output_path / img_id / f"{oid}.clines.npy", clines)
                    # for oid, clines_rgb, clines_seg in zip(oids, clines_rgb_list, clines_seg_list):
                    #     np.save(output_path / img_id / f"{oid}.clines.rgb.npy", clines_rgb)
                    #     np.save(output_path / img_id / f"{oid}.clines.seg.npy", clines_seg)
                        # np.save(output_path / f"{img_id}_{oid}.clines.rgb.npy", clines_rgb)
                        # np.save(output_path / f"{img_id}_{oid}.clines.seg.npy", clines_seg)
                    
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
        "save_figures": False,
        "data_path": "data/webdatasets",
        "dataset_name": "gso_1M",
    }
    #------------------------------------------#
    
    # Create partial function
    extract_clines_partial = partial(extract_clines, config=config)
    
    # Create the output directory if it does not exist
    (Path(config["data_path"]) / (config["dataset_name"]+"_clines_to_remove")).mkdir(exist_ok=True)

    # Get the shard ids
    dataset_path = Path(config["data_path"]) / config["dataset_name"]
    shard_ids = sorted(
        shard.stem for shard in dataset_path.iterdir() if shard.suffix == ".tar"
    )
    
    shard_ids_split = np.array_split(shard_ids, 20)
    
    with multiprocessing.Pool(20) as pool:
        pool.map(extract_clines_partial, shard_ids_split)
    
    # extract_clines_partial(shard_ids[0:1])
