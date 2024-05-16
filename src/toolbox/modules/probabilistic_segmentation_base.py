# Standard libraries
from abc import ABC, abstractmethod

# Third-party libraries
import torch
import torch.nn as nn


class ProbabilisticSegmentationBase(nn.Module, ABC):
    """
    Module that predicts probabilistic segmentation maps from RGB images and binary
    masks.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(
        self,
        rgb_images: torch.Tensor,
        binary_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the module.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, 3, H, W). Values should
                be in the range [0, 255] and of type torch.uint8.
            binary_masks (torch.Tensor): Batch of binary masks (B, 1, H, W). Values should
                be either 0 or 1 and of type torch.float32.

        Returns:
            torch.Tensor: Batch of probabilistic segmentation maps (B, H, W). Values
                are of type torch.float32.
        """
        # Check that the input tensors are of correct dimension, type and shape
        if rgb_images.dim() != 4:
            raise ValueError(
                "Input RGB images tensor is of incorrect shape. "
                f"Expected 4D tensor but got {rgb_images.dim()}D tensor."
            )
        elif binary_masks.dim() != 4:
            raise ValueError(
                "Input binary masks tensor is of incorrect shape. "
                f"Expected 4D tensor but got {binary_masks.dim()}D tensor."
            )
        elif rgb_images.dtype != torch.uint8:
            raise ValueError(
                "Input RGB images tensor is of incorrect type. "
                f"Expected torch.uint8 but got {rgb_images.dtype}."
            )
        elif binary_masks.dtype != torch.float32:
            raise ValueError(
                "Input binary masks tensor is of incorrect type. "
                f"Expected torch.float32 but got {binary_masks.dtype}."
            )
        elif rgb_images.size(1) != 3:
            raise ValueError(
                "Input RGB images tensor is of incorrect shape. "
                f"Expected tensor with 3 channels but got tensor with "
                f"{rgb_images.size(1)} channels."
            )
        elif binary_masks.size(1) != 1:
            raise ValueError(
                "Input binary masks tensor is of incorrect shape. "
                f"Expected tensor with 1 channel but got tensor with "
                f"{binary_masks.size(1)} channels."
            )
        elif rgb_images.size(0) != binary_masks.size(0) or\
            rgb_images.size(2) != binary_masks.size(2) or\
            rgb_images.size(3) != binary_masks.size(3):
            raise ValueError(
                "Input RGB images and binary masks tensors have incompatible shapes. "
                f"Expected tensors of shape {rgb_images.shape} and {binary_masks.shape}"
                " to have the same batch size and spatial dimensions."
            )
        elif binary_masks[
            torch.logical_and(
                binary_masks != 0.0,
                binary_masks != 1.0
            )].numel() > 0:
            raise ValueError(
                "Input binary masks tensor has incorrect values. "
                "Expected tensor with values either 0 or 1."
            )
        
        # Forward pass through the module
        probabilistic_segmentations = self._forward(rgb_images, binary_masks)
        
        # Check that the output tensor is of correct dimension, type and shape
        if probabilistic_segmentations.dim() != 3:
            raise ValueError(
                "Output tensor is of incorrect shape. "
                f"Expected 3D tensor but got {probabilistic_segmentations.dim()}D "
                "tensor."
            )
        elif probabilistic_segmentations.dtype != torch.float32:
            raise ValueError(
                "Output tensor is of incorrect type. "
                f"Expected torch.float32 but got {probabilistic_segmentations.dtype}."
            )
        elif probabilistic_segmentations.size(0) != rgb_images.size(0) or\
            probabilistic_segmentations.size(1) != rgb_images.size(2) or\
            probabilistic_segmentations.size(2) != rgb_images.size(3):
            raise ValueError(
                "Output tensor has incorrect shape. "
                f"Expected tensor of shape {rgb_images.shape} but got "
                f"{probabilistic_segmentations.shape}."
            )

        return probabilistic_segmentations
    
    
    @abstractmethod
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
        pass
