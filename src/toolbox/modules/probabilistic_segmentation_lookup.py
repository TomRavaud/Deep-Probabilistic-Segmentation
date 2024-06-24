# Standard libraries
from typing import Union

# Third-party libraries
import torch
import torch.nn as nn
from torchvision import transforms
from omegaconf import ListConfig

# Custom modules
from toolbox.modules.probabilistic_segmentation_base import (
    ProbabilisticSegmentationBase
)
from toolbox.modules.segmentation_with_histograms_module import (
    SegmentationWithHistograms
)
from toolbox.utils.rgb2hsv_torch import rgb2hsv_torch


class ProbabilisticSegmentationLookup(ProbabilisticSegmentationBase):
    
    def __init__(
        self,
        net: nn.Module = None,
        compile: bool = False,
        color_space: str = "rgb",
        nb_bins: Union[ListConfig, tuple] = (10, 10, 10),
        use_histograms: bool = False,
        output_logits: bool = True,
    ) -> None:
        """Constructor of the class.

        Args:
            net (nn.Module, optional): Neural network module for implicit object
                segmentation. Defaults to None.
            compile (bool, optional): Whether to compile the network. Defaults
                to False.
            color_space (str, optional): Color space of the input images. Defaults to
                "rgb".
            nb_bins (Union[ListConfig, tuple], optional): Number of bins for each color
                channel. Defaults to (10, 10, 10).
            use_histograms (bool, optional): If True, use histograms for implicit
                object segmentation. Defaults to False.
            output_logits (bool, optional): Whether to output logits or probabilities.
                Defaults to True.
        """
        super().__init__()
        
        self._color_space = color_space
        self._nb_bins = tuple(nb_bins)
        
        # Whether to use histograms or a network for implicit object segmentation
        if use_histograms:
            self._segmentation_with_histograms = SegmentationWithHistograms(
                color_space=self._color_space,
                nb_bins=self._nb_bins,
            )
            # No need to normalize the RGB images (identity function)
            self._normalize_transform = lambda x: x
        
        else:
            # Instantiate the network
            # (for implicit object segmentation prediction)
            self._net = net(output_dim=self._nb_bins, output_logits=output_logits)
            
            if compile:
                self._net =\
                    torch.compile(self._net)
            
            self._normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Statistics from ImageNet
                std=[0.229, 0.224, 0.225],
            )
        
        self._implicit_segmentations = None
    
    @staticmethod
    def _masks_by_lookup(
        rgb_images: torch.Tensor,
        implicit_segmentations: torch.Tensor,
        color_space: str = "rgb",
        nb_bins: tuple = (10, 10, 10),
    ) -> torch.Tensor:
        """Compute the probabilistic masks for the input images by looking up the
        implicit segmentations tensor.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, 3, H, W). Values should
                be in the range [0, 1] and of type torch.float32.
            implicit_segmentations (torch.Tensor): Batch of implicit segmentations
                vectors (B, HUE_VALUES). Values should be in the range [0, 1] and of
                type torch.float32.
            color_space (str, optional): Color space of the input images. Defaults to
                "rgb".
            nb_bins (tuple, optional): Number of bins for each color channel. Defaults
                to (10, 10, 10).

        Returns:
            torch.Tensor: Predicted probabilistic masks (B, H, W).
        """
        if color_space == "h":
            
            hsv_images = rgb2hsv_torch(rgb_images)
        
            # Extract the hue channel
            hue_channel = hsv_images[:, 0, :, :]  # assuming hue is the first channel

            # Scale the hue values to match the range of indices in implicit_segmentations
            scaled_hue = (hue_channel * nb_bins[0]).long()
            
            color_values = scaled_hue
            
            # Add a dimension to the implicit segmentations tensor before expanding it
            implicit_segmentations = implicit_segmentations.unsqueeze(-1)

            # Expand the implicit segmentations tensor to match the shape of scaled_hue
            implicit_segmentations = implicit_segmentations.expand(
                -1, -1, color_values.shape[2]
            )

            # Gather values from implicit_segmentations using the scaled hue as indices
            probabilistic_masks = torch.gather(
                implicit_segmentations,
                1,
                color_values,
            )
            
            return probabilistic_masks
            
        elif color_space == "rgb":
            
            color_values = rgb_images

            # Convert the bin sizes to a tensor
            nb_bins = torch.tensor(nb_bins, dtype=torch.long).to(rgb_images.device)

            # Reshape the tensor of bin sizes to match the number of dimensions of the
            # color values tensor
            nb_bins = nb_bins.view(1, -1, 1, 1)

            # Bin the color values
            binned_color_values = ((color_values - 1e-6) * nb_bins).long()

            B, C, H, W = binned_color_values.shape

            # Initialize the probabilistic masks tensor
            probabilistic_masks = torch.empty(
                (B, H, W),
                device=binned_color_values.device,
            )

            # Create the batch indices
            if implicit_segmentations.size(0) == B:
                batch_indices = torch.arange(
                    B,
                    dtype=torch.int,
                    device=binned_color_values.device,
                ).view(B, 1, 1).expand(B, H, W)
            elif implicit_segmentations.size(0) == 1:
                batch_indices = torch.zeros(
                    B,
                    H,
                    W,
                    dtype=torch.int,
                    device=binned_color_values.device,
                )
            else:
                raise ValueError(
                    "The number of implicit segmentations should be either 1 or equal"
                    "to the batch size."
                )

            # Indices for each channel
            a_indices = binned_color_values[:, 0, :, :]
            b_indices = binned_color_values[:, 1, :, :]
            c_indices = binned_color_values[:, 2, :, :]

            # Use torch.gather to fetch the implicit segmentations
            probabilistic_masks = implicit_segmentations[
                batch_indices,
                a_indices,
                b_indices,
                c_indices,
            ]
        
        else:
            raise ValueError(f"Unknown color space: {color_space}")
        
        return probabilistic_masks
        
    def _forward_pixel_segmentation(
        self,
        rgb_images: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the module for pixel segmentation only.
        
        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, C, H, W). Values should
                be in the range [0, 255] and of type torch.uint8.
        
        Returns:
            torch.Tensor: Batch of probabilistic segmentation maps (B, H, W). Values
                are in the range [0, 1] and of type torch.float32.
        """
        # Convert [0, 255] -> [0.0, 1.0]
        rgb_images = rgb_images.to(dtype=torch.float32)
        rgb_images /= 255.0
        
        # Generate the segmentation masks
        probabilistic_masks = self._masks_by_lookup(
            rgb_images,
            self._implicit_segmentations,
            color_space=self._color_space,
            nb_bins=self._nb_bins,
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
        
        # Predict implicit object segmentations using the network or the
        # SegmentationWithHistograms module (histograms + Bayes)
        implicit_segmentation_module =\
            self._net if hasattr(self, "_net")\
                else self._segmentation_with_histograms
        
        self._implicit_segmentations = implicit_segmentation_module(
            input_implicit_segmentation,
        )
        
        # TODO: Add some random noise the RGB images to make the segmentation model more robust
        # Generate the segmentation masks
        probabilistic_masks = self._masks_by_lookup(
            rgb_images,
            self._implicit_segmentations,
            color_space=self._color_space,
            nb_bins=self._nb_bins,
        )
        
        return probabilistic_masks
