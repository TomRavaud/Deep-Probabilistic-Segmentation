# Standard libraries
from typing import Optional

# Third-party libraries
import torch
from torch import nn
from torchvision import transforms
from omegaconf import DictConfig, ListConfig
import numpy as np

# Custom modules
from toolbox.datasets.segmentation_dataset import BatchSegmentationData
from toolbox.datasets.make_sets import make_object_set
from toolbox.modules.contour_rendering_module import ContourRendering
from toolbox.modules.mobile_sam_module import MobileSAM
from toolbox.modules.resnet18_module import ResNet18


class ObjectSegmentationModel(nn.Module):
    """
    TODO: update description
    This module is composed of four parts:
        1. A rendering stage that renders the object in a perturbed pose and extracts
            points along the object contour;
        2. A light Segment Anything Model (MobileSAM) for explicit object segmentation
            alignment;
        
        (1. and 2. can be abandoned if we use ground truth masks as explicit segmentation)
        
        3. A ResNet18 model for implicit object segmentation prediction.
        4. A segmentation mask module for generating segmentation masks from the
            predicted implicit probabilistic object segmentation.
    """
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
        image_size: ListConfig,
        use_gt_masks: bool = True,
        sam_checkpoint: Optional[str] = None,
        object_set_cfg: Optional[DictConfig] = None,
        compile: bool = False,
    ) -> None:
        """
        Initialize an `ObjectSegmentationModel` module.
        """
        super().__init__()
        
        if not use_gt_masks:
            
            # Create the set of objects
            object_set = make_object_set(**object_set_cfg)
        
            # Instantiate the contour rendering module
            # (for rendering 3D objects, and extracting points along objects contour)
            self._contour_rendering_module = ContourRendering(
                object_set=object_set,
                image_size=tuple(image_size),
                debug=True
            )

            # Instantiate the MobileSAM module
            # (for explicit object segmentation alignment)
            self._mobile_sam = MobileSAM(
                sam_checkpoint=sam_checkpoint,
                compile=compile,
            )
            # Freeze the MobileSAM parameters
            for param in self._mobile_sam.parameters():
                param.requires_grad = False
        
        self._use_gt_masks = use_gt_masks
        
        self._probabilistic_segmentation_model = probabilistic_segmentation_model
        
        
    def forward(self, x: BatchSegmentationData) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x (BatchSegmentationData): A batch of segmentation data.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        # Get binary masks
        if self._use_gt_masks:
            binary_masks = x.masks.unsqueeze(1)
        else:
            # Render objects of the batch, extract outer contours points
            contour_points_list = self._contour_rendering_module(x)

            # Predict masks, scores and logits using the MobileSAM model
            mobile_sam_outputs = self._mobile_sam(x, contour_points_list)

            # Stack the masks from the MobileSAM outputs
            binary_masks = torch.stack([
                output["masks"][:, torch.argmax(output["iou_predictions"])]
                for output in mobile_sam_outputs
            ])
        
        # Get RGB images
        rgb_images = x.rgbs
        
        # Compute the probabilistic segmentation masks
        probabilistic_masks = self._probabilistic_segmentation_model(
            rgb_images,
            binary_masks,
        )
        
        return probabilistic_masks


if __name__ == "__main__":
    pass
