# Standard libraries
from typing import Optional

# Third-party libraries
import torch
from torchvision import transforms

# Custom modules
from toolbox.modules.probabilistic_segmentation_base import (
    ProbabilisticSegmentationBase
)
from toolbox.modules.resnet18_module import ResNet18
from toolbox.modules.segmentation_with_histograms_module import (
    SegmentationWithHistograms
)
from toolbox.utils.rgb2hsv_torch import rgb2hsv_torch


class ProbabilisticSegmentationLookup(ProbabilisticSegmentationBase):
    
    HUE_VALUES = 180
    
    def __init__(
        self,
        compile: bool = False,
        use_histograms: bool = False,
        output_logits: bool = True,
    ) -> None:
        """Constructor of the class.

        Args:
            compile (bool, optional): Whether to compile the ResNet18 module. Defaults
                to False.
            use_histograms (bool, optional): If True, use histograms for implicit
                object segmentation. Defaults to False.
            output_logits (bool, optional): Whether to output logits or probabilities.
                Defaults to True.
        """
        super().__init__()
        
        # Whether to use histograms or a ResNet18 for implicit object segmentation
        if use_histograms:
            self._segmentation_with_histograms = SegmentationWithHistograms(
                output_dim=self.HUE_VALUES,
            )
            # No need to normalize the RGB images (identity function)
            self._normalize_transform = lambda x: x
        
        else:
            # Instantiate the ResNet18 module
            # (for implicit object segmentation prediction)
            self._resnet18 = ResNet18(
                output_dim=self.HUE_VALUES,
                nb_input_channels=4,  # 3 RGB channels + 1 mask channel
                output_logits=output_logits,
            )
            
            if compile:
                self._resnet18 =\
                    torch.compile(self._resnet18)
            
            self._normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Statistics from ImageNet
                std=[0.229, 0.224, 0.225],
            )
    
    @staticmethod
    def _masks_by_lookup(
        rgb_images: torch.Tensor,
        implicit_segmentations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the probabilistic masks for the input images by looking up the
        implicit segmentations tensor.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, 3, H, W). Values should
                be in the range [0, 1] and of type torch.float32.
            implicit_segmentations (torch.Tensor): Batch of implicit segmentations
                vectors (B, HUE_VALUES). Values should be in the range [0, 1] and of
                type torch.float32.

        Returns:
            torch.Tensor: Predicted probabilistic masks (B, H, W).
        """
        # Convert RGB images to HSV
        hsv_images = rgb2hsv_torch(rgb_images)
        
        # Extract the hue channel
        hue_channel = hsv_images[:, 0, :, :]  # assuming hue is the first channel
        
        # Scale the hue values to match the range of indices in implicit_segmentations
        scaled_hue = (hue_channel * 180).long()
        
        # Add a dimension to the implicit segmentations tensor before expanding it
        implicit_segmentations = implicit_segmentations.unsqueeze(-1)
        
        # Expand the implicit segmentations tensor to match the shape of scaled_hue
        implicit_segmentations = implicit_segmentations.expand(
            -1, -1, scaled_hue.shape[2]
        )
        
        # Gather values from implicit_segmentations using the scaled hue as indices
        probabilistic_masks = torch.gather(
            implicit_segmentations,
            1,
            scaled_hue,
        )
        
        return probabilistic_masks
    
    def _forward(
        self,
        rgb_images: torch.Tensor,
        binary_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the module.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, C, H, W). Values should
                be in the range [0, 255] and of type torch.uint8.
            binary_masks (torch.Tensor): Batch of binary masks (B, H, W). Values should
                be either 0 or 1 and of type torch.float32.

        Returns:
            torch.Tensor: Batch of probabilistic segmentation maps (B, H, W). Values
                are in the range [0, 1] and of type torch.float32.
        """
        # Convert [0, 255] -> [0.0, 1.0]
        rgb_images = rgb_images.to(dtype=torch.float32)
        rgb_images /= 255.0
        
        # Normalize RGB images
        rgb_images_normalized = self._normalize_transform(rgb_images)
        
        # Concatenate masks and RGB images
        input_implicit_segmentation =\
            torch.cat([rgb_images_normalized, binary_masks], dim=1)
        
        # Predict implicit object segmentations using the ResNet18 model or the
        # SegmentationWithHistograms module (histograms + Bayes)
        implicit_segmentation_module =\
            self._resnet18 if hasattr(self, "_resnet18")\
                else self._segmentation_with_histograms
        
        implicit_segmentations = implicit_segmentation_module(
            input_implicit_segmentation,
        )
        
        # TODO: Add some random noise the RGB images to make the segmentation model more robust
        # Generate the segmentation masks
        probabilistic_masks = self._masks_by_lookup(
            rgb_images,
            implicit_segmentations,
        )
        
        return probabilistic_masks
