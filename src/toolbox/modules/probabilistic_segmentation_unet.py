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
from toolbox.modules.unet_1d_filmed_module import UNet1d


class ProbabilisticSegmentationUNet(nn.Module):
    
    def __init__(
        self,
        net: nn.Module,
        line_segmentation_model: nn.Module,
        apply_color_transformations: bool = False,
    ) -> None:
        """Constructor of the class.
        
        Args:
            net: Network used to predict the context vectors.
            line_segmentation_model: Network used to predict the probabilistic
                segmentation of the contour lines.
            apply_color_transformations (bool, optional): Whether to apply color
                transformations to the input images. You should not use this option
                for inference. Defaults to False.
        """
        super().__init__()
        
        # Attribute to store the context vectors
        self._context_vectors = None
        
        # Network for the context vectors prediction
        self._net = net 
        
        # Network for the probabilistic segmentation of the contour lines
        self._lines_segmentation_model = line_segmentation_model
        
        self._normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Statistics from ImageNet
            std=[0.229, 0.224, 0.225],
        )
        
        if apply_color_transformations:
            # Color transformations to apply to the original image
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

    
    #TODO: to update
    # def forward_line_segmentation(self, rgb_images: torch.Tensor) -> torch.Tensor:
    #     """Forward pass through the pixel segmentation model.

    #     Args:
    #         rgb_images (torch.Tensor): Batch of RGB images (B, 3, H, W). Values should
    #             be in the range [0, 255] and of type torch.uint8.

    #     Returns:
    #         torch.Tensor: Batch of probabilistic segmentation maps (B, H, W). Values
    #             are of type torch.float32.
    #     """
    #      # Convert [0, 255] -> [0.0, 1.0]
    #     rgb_images = rgb_images.to(dtype=torch.float32)
    #     rgb_images /= 255.0
        
    #     # Normalize RGB images
    #     rgb_images_normalized = self._normalize_transform(rgb_images)
        
    #     # Compute the probabilistic masks for the input images
    #     probabilistic_masks = self._apply_pixel_segmentation(
    #         rgb_images_normalized,
    #     )
        
    #     return probabilistic_masks

    def forward(
        self,
        rgb_images: torch.Tensor,
        binary_masks: torch.Tensor,
        clines_rgbs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the module.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, C, H, W). Values should
                be in the range [0, 255] and of type torch.uint8.
            binary_masks (torch.Tensor): Batch of binary masks (B, H, W). Values should
                be either 0 or 1 and of type torch.float32.
            clines_rgbs (torch.Tensor): Batch of RGB contour lines (B, C, L). Values
                should be in the range [0, 255] and of type torch.uint8.

        Returns:
            torch.Tensor: Batch of probabilistic segmentation maps (B, L). Values are
                of type torch.float32.
        """
        #-----------------------------------------------------------------#
        # Context vectors prediction from the RGB images and binary masks #
        #-----------------------------------------------------------------#
        # Convert [0, 255] -> [0.0, 1.0]
        rgb_images = rgb_images.to(dtype=torch.float32)
        rgb_images /= 255.0
        
        # Normalize RGB images
        rgb_images = self._normalize_transform(rgb_images)
        
        # Concatenate masks and RGB images
        input_implicit_segmentation = torch.cat([rgb_images, binary_masks], dim=1)

        # Predict as many context vectors as the number of images in the batch
        self._context_vectors = self._net(input_implicit_segmentation)
        
        
        #-----------------------------------------------------------------#
        # Probabilistic segmentation of the contour lines                 #
        #-----------------------------------------------------------------#
        #TODO: apply color transformations?
        # # Apply color transformations to the RGB lines
        # clines_rgb = self._color_transform(clines_rgb)
        
        # Convert [0, 255] -> [0.0, 1.0]
        clines_rgbs = clines_rgbs.to(dtype=torch.float32)
        clines_rgbs /= 255.0
        
        # Normalize the RGB correspondence lines
        clines_rgbs = self._normalize_transform(clines_rgbs)
        
        nb_lines_per_image = clines_rgbs.size(2)
        
        # Concatenate the lines from all the images to simultaneously predict the
        # probabilistic segmentation of all the lines
        clines_rgbs = clines_rgbs.permute(0, 2, 1, 3)
        clines_rgbs = clines_rgbs.contiguous().view(
            -1,
            clines_rgbs.size(2),
            clines_rgbs.size(3),
        )
        # Save the image
        # import cv2
        # img = clines_rgbs.permute(0, 2, 1).cpu().detach().numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("clines.png", img)
        
        # Compute the probabilistic mask for the lines
        clines_probabilistic_masks = self._lines_segmentation_model(
            clines_rgbs,
            self._context_vectors,
        ).squeeze(1)
        
        # Recover the original shape of the probabilistic masks
        clines_probabilistic_masks = clines_probabilistic_masks.view(
            rgb_images.size(0),
            nb_lines_per_image,
            clines_probabilistic_masks.size(1),
        )
        
        return clines_probabilistic_masks


if __name__ == "__main__":
    pass
