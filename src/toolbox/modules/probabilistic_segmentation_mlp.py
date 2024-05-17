# Standard libraries
from typing import Optional
from functools import partial

# Third-party libraries
import torch
import torch.nn as nn
from torchvision import transforms

# Custom modules
from toolbox.modules.probabilistic_segmentation_base import (
    ProbabilisticSegmentationBase
)
from toolbox.modules.resnet18_module import ResNet18
from toolbox.modules.pixel_segmentation_mlp_module import PixelSegmentationMLP


class ProbabilisticSegmentationMLP(ProbabilisticSegmentationBase):
    
    def __init__(
        self,
        patch_size: int = 5,
        compile: bool = False,
        device: Optional[torch.device] = None,
        output_logits: bool = True,
    ) -> None:
        """Constructor of the class.
        
        Args:
            patch_size (int, optional): Side length of the square patch. Defaults to 5.
            compile (bool, optional): Whether to compile the ResNet18 module. Defaults
                to False.
            device (torch.device, optional): Device on which to run the module. Defaults
                to None.
            output_logits (bool, optional): Whether to output logits or probabilities.
                Defaults to True.
        """
        super().__init__()
        
        # Instantiate the model used to perform pixel-wise segmentation
        self._pixel_segmentation_template = PixelSegmentationMLP(
            patch_size=patch_size,
            nb_channels=3,  # RGB channels
            output_logits=output_logits,
        ).to(device=device)
        
        # Get the number of parameters of the pixel segmentation model
        nb_parameters_template = self._pixel_segmentation_template.nb_parameters
        
        # Instantiate the ResNet18 module
        # (for MLP weights and biases prediction)
        self._resnet18 = ResNet18(
            output_dim=nb_parameters_template,
            nb_input_channels=4,  # 3 RGB channels + 1 mask channel
            output_logits=True,  # True to get the weights and biases of the MLP
        ).to(device=device)
        
        # if inference:
        #     self._resnet18.eval()
        
        if compile:
            self._resnet18 =\
                torch.compile(self._resnet18)
        
        self._normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Statistics from ImageNet
            std=[0.229, 0.224, 0.225],
        )

        self._patch_size = patch_size
        self._device = device
    
    @property
    def device(self) -> torch.device:
        """Return the device on which the module is running.

        Returns:
            torch.device: Device on which the module is running.
        """
        return self._device
    
    @device.setter
    def device(self, device: torch.device) -> None:
        """Set the device on which the module should run.

        Args:
            device (torch.device): Device on which the module should run.
        """
        self._device = device
        
        self._pixel_segmentation_template.to(device=device)
        self._resnet18.to(device=device)
    
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
            mode="constant",
            value=0,
        )

        # Extract the patches from the images
        patches = images.unfold(2, patch_size, 1)\
            .unfold(3, patch_size, 1)

        # Permute the dimensions -> (B, H, W, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)

        # Reshape the patches -> (B, HxW, C, patch_size * patch_size)
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
        # patches = self._unfold(rgb_images)
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
        # Ensure that input tensors are on the same device as the module
        if rgb_images.device != self._device:
            rgb_images = rgb_images.to(device=self._device)
        if binary_masks.device != self._device:
            binary_masks = binary_masks.to(device=self._device)
        
        # Convert [0, 255] -> [0.0, 1.0]
        rgb_images = rgb_images.to(dtype=torch.float32)
        rgb_images /= 255.0
        
        # Normalize RGB images
        rgb_images_normalized = self._normalize_transform(rgb_images)
        
        # Concatenate masks and RGB images
        input_implicit_segmentation =\
            torch.cat([rgb_images_normalized, binary_masks], dim=1)
        
        
        # Predict as much weights and biases sets as the number of images in the batch
        pixel_segmentation_parameters = self._resnet18(input_implicit_segmentation)
        
        # Initialize the output masks tensor
        probabilistic_masks = torch.empty(
            rgb_images.shape[0],
            rgb_images.shape[2],
            rgb_images.shape[3],
        ).to(device=self._device)

        # TODO: Add some random noise the RGB images to make the segmentation model more robust
        # Compute the probabilistic mask for each image in the batch
        for i in range(rgb_images.shape[0]):
            
            # Set the parameters of the template model with the i-th set of parameters
            pixel_segmentation_model = partial(
                self._pixel_segmentation_template,
                parameters=pixel_segmentation_parameters[i],
            )
            
            # Set the mask for the i-th image in the batch
            probabilistic_masks[i] = self._masks_by_model(
                rgb_images[i].unsqueeze(0),
                pixel_segmentation_model,
                patch_size=self._patch_size,
            )
        
        return probabilistic_masks


if __name__ == "__main__":
    
    # Instantiate the model
    probabilistic_segmentation_model = ProbabilisticSegmentationMLP(
        patch_size=5,
        compile=False,
        device=torch.device("cuda:0"),
    )
    
    # Create a random input tensor with appropriate shape
    input_tensor = torch.randint(0, 256, (2, 3, 480, 640)).to(dtype=torch.uint8)
    
    # Create a random binary mask tensor with appropriate shape
    mask_tensor = torch.randint(0, 2, (2, 1, 480, 640)).to(dtype=torch.float32)
    
    # Forward pass
    output = probabilistic_segmentation_model(input_tensor, mask_tensor)
    print("Output shape: ", output.shape)
