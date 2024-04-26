# Standard libraries
from typing import Tuple

# Third-party libraries
import torch
import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.utils import cameras_from_opencv_projection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

# Custom modules
from toolbox.datasets.object_set import RigidObjectSet
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


class ContourRendering(nn.Module):
    """
    Module that renders 3D objects, extract their contours and sample points on them.
    """
    def __init__(
        self,
        object_set: RigidObjectSet,
        image_size: Tuple[int, int],
    ) -> None:
        
        super().__init__()
        
        self._object_set = object_set
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._image_size = image_size
        
        # Set rasterization settings
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,  # To use the faster coarse-to-fine rasterization method
            max_faces_per_bin=None,
        )
        
        self._rasterizer = MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings,
        )
        
        # Set the scaling factor
        self._scale = None
        
        if object_set.scale is not None:
            self._scale = object_set.scale
        else:
            raise NotImplementedError(
                "Scaling the meshes to different scales is not supported."
            )
    
    @staticmethod
    def _generate_valid_contour(mask: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        
        # mask = mask * 255  # For visualization purposes only
        mask = mask.astype(np.uint8)  # Required by cv2.findContours

        contours, hierarchy = cv2.findContours(
            mask,
            # Contours organized in a 2-level hierarchy (external and internal contours)
            cv2.RETR_CCOMP,
            # All the boundary points are stored
            cv2.CHAIN_APPROX_NONE,
        )
        
        # print("Number of contours:", len(contours))
        
        return contours, hierarchy
    
    # TODO: change input to contain only the necessary data and the output to be more
    # explicit
    def forward(self, x: BatchSegmentationData) -> Tuple[Tuple, np.ndarray]:
        
        # Create the set of labels to filter the objects
        keep_labels = set(x.object_datas[i].label for i in range(x.batch_size))
        
        # Create a new temporary object set with only the objects of the batch
        object_set = self._object_set.filter_objects(keep_labels)
        
        # Get the paths to the meshes of the objects
        mesh_paths = [object.mesh_path for object in object_set]
        
        # NOTE: Preload the meshes to avoid loading them at each forward pass?
        # Load the meshes (without textures)
        meshes = load_objs_as_meshes(files=mesh_paths,
                                     load_textures=False,
                                     device=self._device)
        
        # In-place scaling of the vertices
        if self._scale is not None:
            meshes.scale_verts_(object_set.scale)
        
        
        ### Rasterize only the first mesh ###
        
        # Find the index of the mesh that corresponds to the object in the first
        # observation of the batch
        obj_label = x.object_datas[0].label
        
        idx = None
        for i, object in enumerate(object_set):
            if object.label == obj_label:
                idx = i
                break
        
        # Get the transformation matrix of the object
        TCO = x.TCO[0]
        
        # Rotation matrix and translation vector
        R = TCO[:3, :3].unsqueeze(0)
        tvec = TCO[:3, 3].unsqueeze(0)
        # Camera matrix
        K = x.K[0].unsqueeze(0)
        
        # Set the camera
        camera = cameras_from_opencv_projection(
            R=R,
            tvec=tvec,
            camera_matrix=K,
            image_size=torch.Tensor(x.image_size).unsqueeze(0),
        ).cuda()
        
        
        # Generate the depth map
        depth_map = self._rasterizer(
            meshes[idx],
            cameras=camera,
        ).zbuf
        
        # Create a mask from the depth map
        mask = (depth_map > 0).type(torch.float32)
        mask = mask.squeeze().cpu().numpy()
        
        
        # Get the contour points
        contour_points, hierarchy = ContourRendering._generate_valid_contour(
            mask,
        )
        print("Hierarchy:", hierarchy)
        
        
        ###########################################################################
        # Debugging
        ###########################################################################
        print("Name of the object:", x.object_datas[0].label)
        
        plt.figure()
        plt.imshow(depth_map.squeeze().cpu().numpy())
        
        # Get the bounding box
        bbox = x.bboxes[0].numpy()
        bbox = bbox.astype(int)
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='m',
            facecolor='none',
        )
        
        _, ax = plt.subplots()
        
        ax.imshow(x.rgbs[0].permute(1, 2, 0).cpu().numpy())
        ax.add_patch(rect)
        
        # Mask the image
        plt.imshow(mask, alpha=0.3)
        
        plt.show()
        
        
        # Draw the contours
        contour = contour_points[0]
        mask = mask.astype(np.uint8)
        mask *= 255
        
        cv2.drawContours(
            image=mask,
            contours=[contour],
            contourIdx=-1,
            color=120,
            thickness=5,
        )
        
        # Visualize the mask
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        ###########################################################################
        
        return contour_points, hierarchy
