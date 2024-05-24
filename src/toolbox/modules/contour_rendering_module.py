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
        debug: bool = True,
    ) -> None:
        
        super().__init__()
        
        self._object_set = object_set
        
        self._image_size = image_size
        self._debug = debug
        
        # Set rasterization settings
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            # Different from 0 to use the faster coarse-to-fine rasterization method
            bin_size=None,
            max_faces_per_bin=None,
        )
        
        self._rasterizer = MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings,
        )
        
        if not self._debug:
            
            # Get the paths to the meshes of the objects
            mesh_paths = [object.mesh_path for object in object_set]

            # Load the meshes (without textures)
            self._meshes = load_objs_as_meshes(files=mesh_paths,
                                               load_textures=False,
            )

            # In-place scaling of the vertices
            if object_set.scale is not None:
                self._meshes.scale_verts_(object_set.scale)
            else:
                raise NotImplementedError(
                    "Scaling the meshes to different scales is not supported."
                )
    
    @staticmethod
    def _generate_valid_contour(masks: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        
        # mask = mask * 255  # For visualization purposes only
        masks = masks.astype(np.uint8)  # Required by cv2.findContours
        
        contours_list = []
        
        for i in range(masks.shape[0]):

            contours, hierarchy = cv2.findContours(
                masks[i],
                # Contours organized in a 2-level hierarchy
                # (external and internal contours)
                # cv2.RETR_CCOMP,
                # Retrieve only the external contours
                cv2.RETR_EXTERNAL,
                # All the boundary points are stored
                cv2.CHAIN_APPROX_NONE,
            )
            # print("Hierarchy:", hierarchy)
            # print("Number of contours:", len(contours))
            
            contours_list.append([contour.reshape(-1, 2) for contour in contours])
        
        return contours_list
    
    
    @torch.no_grad()
    def forward(self, x: BatchSegmentationData) -> Tuple:
        
        if self._debug:
            
            # Create the set of labels to filter the objects
            keep_labels = set(x.object_datas[i].label for i in range(x.batch_size))

            # Create a new temporary object set with only the objects of the batch
            object_set = self._object_set.filter_objects(keep_labels)

            # Get the paths to the meshes of the objects
            mesh_paths = [object.mesh_path for object in object_set]

            # Load the meshes (without textures)
            meshes = load_objs_as_meshes(files=mesh_paths,
                                         load_textures=False,
            )

            # In-place scaling of the vertices
            if self._object_set.scale is not None:
                meshes.scale_verts_(self._object_set.scale)
            else:
                raise NotImplementedError(
                    "Scaling the meshes to different scales is not supported."
                )
        else:
            object_set = self._object_set
            meshes = self._meshes
        
        meshes = meshes.to(device=x.rgbs.device)
        
        # Get the indexes of the objects in the object set that correspond to the
        # objects in the batch
        batch_objects_idx = [
            object_set.get_id_from_label(obj_data.label) for obj_data in x.object_datas
        ]
        
        # Apply the perturbation to the ground truth pose
        TCO = torch.bmm(x.TCO, x.DTO)
        
        # Rotation matrices and translation vectors
        R = TCO[:, :3, :3]
        tvec = TCO[:, :3, 3]
        
        # Set the cameras
        cameras = cameras_from_opencv_projection(
            R=R,
            tvec=tvec,
            camera_matrix=x.K,
            image_size=torch.Tensor(x.image_size).unsqueeze(0),
        ).to(device=x.rgbs.device)
        
        # Generate the depth map
        depth_maps = self._rasterizer(
            meshes[batch_objects_idx],
            cameras=cameras,
        ).zbuf[..., 0]
        
        # Create masks from the depth maps, send them to the CPU and convert them to
        # numpy arrays for OpenCV processing
        masks = (depth_maps > 0).type(torch.float32).cpu().numpy()
        
        # Get the contour points
        contour_points_list = ContourRendering._generate_valid_contour(
            masks,
        )
        
        
        ###########################################################################
        # Debugging
        ###########################################################################
        # print("Name of the object:", x.object_datas[0].label)
        
        # plt.figure()
        # plt.imshow(depth_maps[0].cpu().numpy())
        
        # # Get the bounding box
        # bbox = x.bboxes[0].numpy()
        # bbox = bbox.astype(int)
        
        # rect = patches.Rectangle(
        #     (bbox[0], bbox[1]),
        #     bbox[2] - bbox[0],
        #     bbox[3] - bbox[1],
        #     linewidth=2,
        #     edgecolor='m',
        #     facecolor='none',
        # )
        
        # _, ax = plt.subplots()
        
        # ax.imshow(x.rgbs[0].permute(1, 2, 0).cpu().numpy())
        # ax.add_patch(rect)
        
        # # Mask the image
        # plt.imshow(masks[0], alpha=0.3)
        
        # plt.show()
        
        # # Draw the contours
        # mask = masks[0].astype(np.uint8)
        # mask *= 255
        
        # cv2.drawContours(
        #     image=mask,
        #     contours=contour_points_list[0],
        #     contourIdx=-1,
        #     color=120,
        #     thickness=5,
        # )
        
        # # Visualize the mask
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(0)
        ###########################################################################
        
        return contour_points_list
