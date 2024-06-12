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

# Custom modules
from toolbox.datasets.object_set import RigidObjectSet
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


class MaskRendering(nn.Module):
    """
    Module that renders 3D objects as binary masks.
    """
    def __init__(
        self,
        object_set: RigidObjectSet,
        image_size: Tuple[int, int],
        debug: bool = False,
    ) -> None:
        """Constructor.

        Args:
            object_set (RigidObjectSet): Object set containing the objects to render.
            image_size (Tuple[int, int]): Size of the rendered masks.
            debug (bool, optional): Flag to enable debug mode. If False, all the meshes
                are loaded and scaled at the beginning. Defaults to False.

        Raises:
            NotImplementedError: Scaling the meshes to different scales is not
                supported.
        """
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
    
    @torch.no_grad()
    def forward(self, x: BatchSegmentationData) -> torch.Tensor:
        """Forward pass.

        Args:
            x (BatchSegmentationData): A batch of segmentation data.

        Raises:
            NotImplementedError: Scaling the meshes to different scales is not
                supported.

        Returns:
            torch.Tensor: A tensor of masks.
        """
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
        
        # Rotation matrices and translation vectors
        R = x.TCO[:, :3, :3]
        tvec = x.TCO[:, :3, 3]
        
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
        masks = (depth_maps > 0).type(torch.float32)
        
        return masks
