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
        sam_checkpoint: str,
        object_set_cfg: Optional[DictConfig] = None,
        compile: bool = False,
    ) -> None:
        """
        Initialize an `ObjectSegmentationModel` module.
        """
        super().__init__()
        
        # Create the set of objects
        object_set = make_object_set(**object_set_cfg)
        
        image_size = tuple(image_size)
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Statistics from ImageNet
                std=[0.229, 0.224, 0.225],
        )
        
        # Instantiate the contour rendering module
        # (for rendering 3D objects, and extracting points along objects contour)
        self._contour_rendering_module = ContourRendering(
            object_set=object_set,
            image_size=image_size,
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
        
        # Instantiate the ResNet18 module
        # (for implicit object segmentation prediction)
        self._resnet18 = ResNet18(
            output_dim=180,  # Hue values
            nb_input_channels=4,  # 3 RGB channels + 1 mask channel
        ).to(device=self._device)
        
        if compile:
            self._resnet18 =\
                torch.compile(self._resnet18)
        
        # Instantiate the segmentation mask module
        # (for segmentation mask computation)
        self._segmentation_mask_module = SegmentationMask()

        if compile:
            self._segmentation_mask_module =\
                torch.compile(self._segmentation_mask_module)


    def forward(self, x: BatchSegmentationData) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x (BatchSegmentationData): A batch of segmentation data.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        # Render objects of the batch, extract outer contours points
        contour_points_list = self._contour_rendering_module(x)
        
        # Predict masks, scores and logits using the MobileSAM model
        mobile_sam_outputs = self._mobile_sam(x, contour_points_list)
        
        # Stack the masks from the MobileSAM outputs
        masks = torch.stack([
            output["masks"][torch.argmax(output["iou_predictions"])]
            for output in mobile_sam_outputs
        ])
        
        # Send images and masks to the device
        x.rgbs = x.rgbs.to(device=self._device)
        masks = masks.to(device=self._device)
        
        # Range [0, 255] -> [0, 1]
        x.rgbs = x.rgbs.to(dtype=torch.float32)
        x.rgbs /= 255.0
        
        # Normalize the RGB images
        rgbs_normalized = self._normalize_transform(x.rgbs)
        
        # Combine masks and RGB images
        input_resnet = torch.cat([rgbs_normalized, masks], dim=1)
        
        # Predict implicit object segmentations using the ResNet18 model
        implicit_segmentations = self._resnet18(input_resnet)
        
        # Generate the segmentation masks
        segmentation_masks = self._segmentation_mask_module(
            x.rgbs,
            implicit_segmentations,
        )
        
        return segmentation_masks


if __name__ == "__main__":
    pass
