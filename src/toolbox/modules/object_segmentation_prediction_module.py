# Standard libraries
from dataclasses import dataclass
from typing import Optional
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third party libraries
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# Custom modules
from toolbox.modules.resnet18_module import ResNet18
from toolbox.modules.mobile_sam_module import MobileSAM
from toolbox.modules.segmentation_with_histograms_module import (
    SegmentationWithHistograms
)
from toolbox.modules.probabilistic_segmentation_lookup import ProbabilisticSegmentationLookup
from toolbox.modules.probabilistic_segmentation_mlp import ProbabilisticSegmentationMLP


@dataclass
class BatchInferenceData:
    """
    Dataclass for the input data of the ObjectSegmentationPredictionModel.
    """
    # Batch of RGB images
    # Shape: (B, C, H, W)
    # Range: [0, 255]
    # Data type: torch.uint8
    rgbs: torch.Tensor
    
    # One element per image in the batch. An element is a list of contours,
    # where each contour is a numpy array of shape (N, 2)
    contour_points_list: list


class ObjectSegmentationPredictionModel(nn.Module):
    """
    Module that predicts object segmentations using the MobileSAM and ResNet18
    pre-trained models.
    """
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
        compile: bool = False,
    ) -> None:
        """Constructor of the ObjectSegmentationPredictionModel.

        Args:
            probabilistic_segmentation_model (nn.Module): Model that predicts
                probabilistic segmentations.
            compile (bool, optional): Whether to compile parts of the model.
                Defaults to False.
        """
        super().__init__()
        
        # Instantiate the MobileSAM module
        # (for explicit object segmentation alignment)
        self._mobile_sam = MobileSAM(
            sam_checkpoint="../weights/mobile_sam.ckpt",
            compile=compile,
        )
        self._binary_masks = None
        
        self._probabilistic_segmentation_model = probabilistic_segmentation_model
    
    @property
    def binary_masks(self) -> Optional[torch.Tensor]:
        """Getter for the binary masks from the MobileSAM model.

        Returns:
            torch.Tensor: Binary masks from the MobileSAM model.
        """
        return self._binary_masks
    
    @torch.no_grad()
    def forward(
        self,
        x: BatchInferenceData,
        pixel_segmentation_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (BatchInferenceData): Input data for the model.
            pixel_segmentation_only (bool, optional): Whether to use the previous
                segmentation model to predict the pixel segmentation. Defaults to False.

        Returns:
            torch.Tensor: Predicted segmentation masks.
        """
        # Get RGB images
        rgb_images = x.rgbs
        
        if pixel_segmentation_only:
            
            self._binary_masks = None
            
            probabilistic_masks =\
                self._probabilistic_segmentation_model.forward_pixel_segmentation(
                    rgb_images,
                )
            return probabilistic_masks
        
        else:
            # Predict masks, scores and logits using the MobileSAM model
            mobile_sam_outputs = self._mobile_sam(x, x.contour_points_list)

            # Stack the masks from the MobileSAM outputs
            self._binary_masks = torch.stack([
                output["masks"][:, torch.argmax(output["iou_predictions"])]
                for output in mobile_sam_outputs
            ])
            
            # Compute the probabilistic segmentation masks
            probabilistic_masks = self._probabilistic_segmentation_model(
                rgb_images,
                self._binary_masks,
            )

            return probabilistic_masks
    

class ObjectSegmentationPredictionModule(nn.Module):
    """
    Module that predicts object segmentations using the
    ObjectSegmentationPredictionModel.
    """
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
        compile: bool = False,
    ) -> None:
        """
        Constructor of the ObjectSegmentationPredictionModule.
        
        Args:
            probabilistic_segmentation_model (nn.Module): Model that predicts
                probabilistic segmentations.
            compile (bool, optional): Whether to compile parts of the model.
                Defaults to False.
        """
        super().__init__()
        
        self._model = ObjectSegmentationPredictionModel(
            probabilistic_segmentation_model=probabilistic_segmentation_model,
            compile=compile,
        )
    
    @property
    def model(self) -> ObjectSegmentationPredictionModel:
        """Getter for the ObjectSegmentationPredictionModel.

        Returns:
            ObjectSegmentationPredictionModel: ObjectSegmentationPredictionModel.
        """
        return self._model
    
    @torch.no_grad()
    def forward(
        self,
        x: BatchInferenceData,
        pixel_segmentation_only: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (BatchInferenceData): Input data for the model.
            pixel_segmentation_only (bool, optional): Whether to use the previous
                segmentation model to predict the pixel segmentation. Defaults to False.

        Returns:
            torch.Tensor: Predicted segmentation masks.
        """
        return self._model(x, pixel_segmentation_only)


if __name__ == "__main__":
    
    # Dummy input
    dummy_input = BatchInferenceData(
        rgbs=torch.rand((1, 3, 480, 640)).to(device="cuda"),
        contour_points_list=[
            # First example of the batch
            [np.array([[0, 0], [50, 50]]),],
            # Second example of the batch...
        ],
    )
    
    prediction_module = ObjectSegmentationPredictionModule()
    
    # Perform a forward pass
    prediction = prediction_module(dummy_input)
    
    print(prediction.shape)
