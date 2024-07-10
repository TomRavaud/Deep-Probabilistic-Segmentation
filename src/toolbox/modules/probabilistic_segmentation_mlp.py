# Standard libraries
from functools import partial
from typing import Union, List

# Third-party libraries
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from omegaconf import ListConfig

# Custom modules
from toolbox.modules.probabilistic_segmentation_base import (
    ProbabilisticSegmentationBase
)
from toolbox.modules.pixel_segmentation_mlp_module import PixelSegmentationMLP
from toolbox.utils.rgb2hsv_torch import rgb2hsv_torch


class ProbabilisticSegmentationMLP(ProbabilisticSegmentationBase):
    
    def __init__(
        self,
        net_cls,
        patch_size: int = 5,
        mlp_hidden_dims: Union[List[int], ListConfig] = [64, 32, 16],
        apply_color_transformations: bool = False,
        compile: bool = False,
        output_logits: bool = True,
    ) -> None:
        """Constructor of the class.
        
        Args:
            net_cls: Partial class to instantiate the network used to predict the
                weights and biases of the MLP. Its arguments are supposed to be already
                set, except for the output dimension which depends on the number of
                parameters of the MLP.
            patch_size (int, optional): Side length of the square patch. Defaults to 5.
            mlp_hidden_dims (Union[List[int], ListConfig], optional): Number of hidden
                units per layer in the MLP. Defaults to [64, 32, 16].
            apply_color_transformations (bool, optional): Whether to apply color
                transformations to the input images. You should not use this option
                for inference. Defaults to False.
            compile (bool, optional): Whether to compile the network. Defaults
                to False.
            output_logits (bool, optional): Whether to output logits or probabilities.
                Defaults to True.
        """
        super().__init__()
        
        # Instantiate the model used to perform pixel-wise segmentation
        self._pixel_segmentation_template = PixelSegmentationMLP(
            patch_size=patch_size,
            nb_channels=3,  # RGB channels
            hidden_dims=list(mlp_hidden_dims),
            output_logits=output_logits,
        )
        
        # Get the number of parameters of the pixel segmentation model
        nb_parameters_template = self._pixel_segmentation_template.nb_parameters
        
        # Attribute to store the parameters of the pixel segmentation model
        self._pixel_segmentation_parameters = None
        
        # Instantiate the network
        # (for MLP weights and biases prediction)
        self._net = net_cls(output_dim=(nb_parameters_template,)) 
        
        if compile:
            self._net =\
                torch.compile(self._net)
        
        self._normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Statistics from ImageNet
            std=[0.229, 0.224, 0.225],
        )
        
        if apply_color_transformations:
            # Color transformations to apply to the original image
            # so that the cnn and mlp do not see the exact same input
            # (makes the model more robust to minor color changes between frames)
            self._color_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                ),
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 2.0)
                ),
            ])
        else:
            self._color_transform = lambda x: x

        self._patch_size = patch_size
    
    @staticmethod
    def _images_to_patches(images: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
        """Extract patches from the RGB images.
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W).
            patch_size (int, optional): Side length of the square patch. Defaults to 5.
        Returns:
            torch.Tensor: Batch of image patches (B, HxW, C, patch_size, patch_size).
        """
        # Check that the input tensor is of the correct dimension
        if images.dim() != 4:
            raise ValueError(
                "Input images tensor is of incorrect shape. "
                f"Expected 4D tensor but got {images.dim()}D tensor."
            )

        # Pad the RGB images in order to generate 1 patch per pixel
        padding_size = patch_size // 2
        images = nn.functional.pad(
            images,
            (padding_size,) * 4,
            mode="replicate",
        )

        # Extract the patches from the images
        patches = images.unfold(2, patch_size, 1)\
            .unfold(3, patch_size, 1)

        # Permute the dimensions -> (B, H, W, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)

        # Reshape the patches -> (B, HxW, C, patch_size, patch_size)
        patches = patches.contiguous().view(
            images.shape[0],
            -1,
            images.shape[1],
            patch_size,
            patch_size,
        )

        return patches
    
    @staticmethod
    def _masks_by_model(
        images: torch.Tensor,
        segmentation_model: nn.Module,
        patch_size: int = 5,
    ) -> torch.Tensor:
        """Compute the probabilistic masks for the input images by applying a
        segmentation model pixel-wise.

        Args:
            images (torch.Tensor): Batch of images (B, C, H, W).
            segmentation_model (nn.Module): Segmentation model to use for pixel-wise
                segmentation.
            patch_size (int, optional): Side length of the square patch. Defaults to 5.

        Returns:
            torch.Tensor: Predicted probabilistic masks (B, H, W).
        """
        # Extract the patches from the images
        patches = ProbabilisticSegmentationMLP._images_to_patches(images, patch_size)
        
        # Predict the probability of each pixel being part of the foreground
        pixel_probabilities = segmentation_model(patches)
        
        # Reshape the pixel probabilities to form the segmentation masks (B, H, W)
        probabilistic_masks = pixel_probabilities.view(
            images.size(0),
            images.size(2),
            images.size(3),
        )
        
        return probabilistic_masks
    
    def _apply_pixel_segmentation(
        self,
        rgb_images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the probabilistic masks for the input images.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, C, H, W).

        Returns:
            torch.Tensor: Batch of probabilistic segmentation maps (B, H, W).
        """
        # Initialize the output masks tensor
        probabilistic_masks = torch.empty(
            rgb_images.shape[0],
            rgb_images.shape[2],
            rgb_images.shape[3],
            device=rgb_images.device,
        )
        
        # Ensure that the pixel segmentation parameters have been set
        if self._pixel_segmentation_parameters is None:
            raise ValueError(
                "Pixel segmentation parameters have not been set. "
                "Please run the forward method first."
            )
        
        # Compute the probabilistic mask for each image in the batch
        for i in range(rgb_images.shape[0]):
            
            # Set the parameters of the template model with the i-th set of parameters
            # if available, otherwise use the first set of parameters
            parameters = self._pixel_segmentation_parameters[i] \
                if len(self._pixel_segmentation_parameters) > i \
                else self._pixel_segmentation_parameters[0]
            
            # Create a partial function with the parameters
            pixel_segmentation_model = partial(
                self._pixel_segmentation_template,
                parameters=parameters,
            )
            
            # Set the mask for the i-th image in the batch
            probabilistic_masks[i] = self._masks_by_model(
                rgb_images[i].unsqueeze(0),
                pixel_segmentation_model,
                patch_size=self._patch_size,
            )
        
        return probabilistic_masks
    
    def _forward_pixel_segmentation(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pixel segmentation model.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, 3, H, W). Values should
                be in the range [0, 255] and of type torch.uint8.

        Returns:
            torch.Tensor: Batch of probabilistic segmentation maps (B, H, W). Values
                are of type torch.float32.
        """
         # Convert [0, 255] -> [0.0, 1.0]
        rgb_images = rgb_images.to(dtype=torch.float32)
        rgb_images /= 255.0
        
        # Normalize RGB images
        rgb_images_normalized = self._normalize_transform(rgb_images)
        
        # Compute the probabilistic masks for the input images
        probabilistic_masks = self._apply_pixel_segmentation(
            rgb_images_normalized,
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
        # Apply color transformations to the RGB images
        rgb_images_transformed = self._color_transform(rgb_images)
        
        # Convert [0, 255] -> [0.0, 1.0]
        rgb_images = rgb_images.to(dtype=torch.float32)
        rgb_images /= 255.0
        rgb_images_transformed = rgb_images_transformed.to(dtype=torch.float32)
        rgb_images_transformed /= 255.0
        
        # Normalize RGB images
        rgb_images_normalized = self._normalize_transform(rgb_images)
        rgb_images_transformed_normalized =\
            self._normalize_transform(rgb_images_transformed)
        
        # Concatenate masks and RGB images
        input_implicit_segmentation =\
            torch.cat([rgb_images_normalized, binary_masks], dim=1)

        # Predict as many weights and biases sets as the number of images in the batch
        self._pixel_segmentation_parameters =\
            self._net(input_implicit_segmentation)
        
        # Compute the probabilistic masks for the input (transformed) images
        probabilistic_masks = self._apply_pixel_segmentation(
            rgb_images_transformed_normalized,
        )
        
        return probabilistic_masks


if __name__ == "__main__":
    
    # Instantiate the model
    probabilistic_segmentation_model = ProbabilisticSegmentationMLP(
        patch_size=5,
        compile=False,
    )
    
    # Create a random input tensor with appropriate shape
    input_tensor = torch.randint(0, 256, (2, 3, 480, 640)).to(dtype=torch.uint8)
    
    # Create a random binary mask tensor with appropriate shape
    mask_tensor = torch.randint(0, 2, (2, 1, 480, 640)).to(dtype=torch.float32)
    
    # Forward pass
    output = probabilistic_segmentation_model(input_tensor, mask_tensor)
    print("Output shape: ", output.shape)
