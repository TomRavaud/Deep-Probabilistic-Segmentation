# Standard libraries
from typing import Optional, Tuple

# Third-party libraries
import torch
from torch import nn
from omegaconf import DictConfig, ListConfig

# Custom modules
from toolbox.datasets.segmentation_dataset import BatchSegmentationData
from toolbox.datasets.make_sets import make_object_set
from toolbox.modules.contour_rendering_module import ContourRendering
from toolbox.modules.mobile_sam_module import MobileSAM
from toolbox.modules.resnet18_module import ResNet18
from toolbox.modules.segmentation_mask_module import SegmentationMask


class ObjectSegmentationModel(nn.Module):
    """
    This module is composed of four parts:
        1. A rendering stage that renders the object in a perturbed pose and extracts
            points along the object contour;
        2. A light Segment Anything Model (MobileSAM) for explicit object segmentation
            alignment;
        3. A ResNet18 model for implicit object segmentation prediction.
        4. A segmentation mask module for generating segmentation masks from the
            predicted implicit probabilistic object segmentation.
    """
    def __init__(
        self,
        image_size: ListConfig,
        object_set_cfg: Optional[DictConfig] = None,
    ) -> None:
        """
        Initialize an `ObjectSegmentationModel` module.
        """
        super().__init__()
        
        # Create the set of objects
        object_set = make_object_set(**object_set_cfg)
        
        image_size = tuple(image_size)
        
        # Instantiate the contour rendering module
        # (for rendering 3D objects, and extracting points along objects contour)
        self._contour_rendering = ContourRendering(
            object_set=object_set,
            image_size=image_size,
        )

        # Instantiate the MobileSAM module
        # (for explicit object segmentation alignment)
        self._mobile_sam = MobileSAM()
        
        # Instantiate the ResNet18 module
        # (for implicit object segmentation prediction)
        # self._resnet18 = ResNet18()
        
        # Instantiate the segmentation mask module
        # (for segmentation mask computation)
        # self._segmentation_mask = SegmentationMask()


    def forward(self, x: BatchSegmentationData) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        # TODO: contour rendering forward pass should not take the full batch
        # segmentation data as input, but only the necessary data
        contour_points, hierarchy = self._contour_rendering(x)
        
        # TODO: do not use the full batch data as input
        # MobileSAM
        mask = self._mobile_sam(x, contour_points, hierarchy)
        
        
        return x


if __name__ == "__main__":
    _ = ObjectSegmentationModel()
