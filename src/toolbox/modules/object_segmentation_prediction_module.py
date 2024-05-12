# Standard libraries
from dataclasses import dataclass
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
from toolbox.modules.segmentation_mask_module import SegmentationMask
from toolbox.modules.segmentation_with_histograms_module import (
    SegmentationWithHistograms
)


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
        use_histograms: bool = False,
        compile: bool = False,
    ) -> None:
        """Constructor of the ObjectSegmentationPredictionModel.

        Args:
            use_histograms (bool, optional): Whether to use histograms for
                implicit object segmentation prediction. Defaults to False.
            compile (bool, optional): Whether to compile parts of the model.
                Defaults to False.
        """
        super().__init__()
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        # Instantiate the MobileSAM module
        # (for explicit object segmentation alignment)
        self._mobile_sam = MobileSAM(
            sam_checkpoint="../weights/mobile_sam.pt",
            compile=compile,
        )
        
        if use_histograms:
            self._segmentation_with_histograms = SegmentationWithHistograms(
                output_dim=180,  # Hue values
            )
            # No need to normalize the RGB images (identity function)
            self._normalize_transform = lambda x: x
        
        else:
            # Instantiate the ResNet18 module
            # (for implicit object segmentation prediction)
            self._resnet18 = ResNet18(
                output_dim=180,  # Hue values
                nb_input_channels=4,  # 3 RGB channels + 1 mask channel
            ).to(device=self._device)
            self._resnet18.eval()
            
            if compile:
                self._resnet18 =\
                    torch.compile(self._resnet18)
            
            self._normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Statistics from ImageNet
                std=[0.229, 0.224, 0.225],
            )
        
        # Instantiate the segmentation mask module
        # (for segmentation mask computation)
        self._segmentation_mask_module = SegmentationMask()
        if compile:
            self._segmentation_mask_module =\
                torch.compile(self._segmentation_mask_module)
    
    @torch.no_grad()
    def forward(self, x: BatchInferenceData) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (BatchInferenceData): Input data for the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted segmentation masks and
                masks from the MobileSAM model.
        """
        # Predict masks, scores and logits using the MobileSAM model
        mobile_sam_outputs = self._mobile_sam(x, x.contour_points_list)
        
        # Stack the masks from the MobileSAM outputs
        masks = torch.stack([
            output["masks"][:, torch.argmax(output["iou_predictions"])]
            for output in mobile_sam_outputs
        ])
        
        rgbs = x.rgbs
        
        # Send images and masks to the device
        rgbs = rgbs.to(device=self._device)
        masks = masks.to(device=self._device)
        
        # Range [0, 255] -> [0, 1]
        rgbs = rgbs.to(dtype=torch.float32)
        rgbs /= 255.0
        
        # Normalize the RGB images
        rgbs_normalized = self._normalize_transform(rgbs)
        
        # Combine masks and RGB images
        input_implicit_segmentation = torch.cat([rgbs_normalized, masks], dim=1)
        
        # Predict implicit object segmentations using the ResNet18 model or the
        # SegmentationWithHistograms module (histograms + Bayes)
        implicit_segmentation_module =\
            self._resnet18 if hasattr(self, "_resnet18")\
                else self._segmentation_with_histograms
        
        implicit_segmentations = implicit_segmentation_module(
            input_implicit_segmentation,
        )
        
        # Generate the segmentation masks
        segmentation_masks = self._segmentation_mask_module(
            rgbs,
            implicit_segmentations,
        )
        
        return segmentation_masks, masks
        

class ObjectSegmentationPredictionModule(nn.Module):
    """
    Module that predicts object segmentations using the
    ObjectSegmentationPredictionModel.
    """
    def __init__(self, use_histograms: bool = False, compile: bool = False) -> None:
        """
        Constructor of the ObjectSegmentationPredictionModule.
        """
        super().__init__()
        
        self._model = ObjectSegmentationPredictionModel(
            use_histograms=use_histograms,
            compile=compile,
        )
    
    @torch.no_grad()
    def forward(self, x: BatchInferenceData) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (BatchInferenceData): Input data for the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted segmentation masks and
                masks from the MobileSAM model.
        """
        return self._model(x)


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
