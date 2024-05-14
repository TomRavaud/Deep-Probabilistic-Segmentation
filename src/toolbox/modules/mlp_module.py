import torch
import torch.nn as nn
import torchinfo


class PixelSegmentation(nn.Module):
    """
    Module that predicts the probability of pixels in an image being part of the
    foreground based on the RGB values of the patches centered at the pixels.
    """
    def __init__(self, patch_size: int = 5, nb_channels: int = 3) -> None:
        """Constructor of the class.

        Args:
            patch_size (int, optional): Side length of the square patch. Defaults to 5.
            nb_channels (int, optional): Number of channels in the input tensor.
                Defaults to 3.
        """
        super(PixelSegmentation, self).__init__()
        
        # Define the architecture of the network
        self._mlp = nn.Sequential(
            nn.Linear(patch_size ** 2 * nb_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self._patch_size = patch_size
        self._nb_channels = nb_channels

    def forward(self, image_patches: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            image_patches (torch.Tensor): Batch of image patches
                (B, nb patches, C, patch_size, patch_size).

        Raises:
            ValueError: If the input tensor if of incorrect dimension
            ValueError: If the input tensor if of incorrect shape

        Returns:
            torch.Tensor: Predictions of the module. 
        """
        # Check that the input tensor is of the correct dimension and shape
        if image_patches.dim() != 5:
            raise ValueError(
                "Input tensor is of incorrect shape. "
                f"Expected 5D tensor but got {len(image_patches.shape)}D tensor."
            )
        elif image_patches.shape[2:] != (self._nb_channels,
                                         self._patch_size,
                                         self._patch_size):
            raise ValueError(
                "Input tensor is of incorrect shape. Expected tensor of shape "
                + str((
                    image_patches.shape[0],
                    image_patches.shape[1],
                    self._nb_channels,
                    self._patch_size,
                    self._patch_size))
                + f" but got {tuple(image_patches.shape)}."
            )
        
        # Flatten the input tensor
        patches_flattened = image_patches.view(
            image_patches.size(0),
            image_patches.size(1),
            -1,
        )
        
        return self._mlp(patches_flattened)


# TODO: integrate with the other SegmentationMask module
class SegmentationMask(nn.Module):
    """
    Module that predicts probabilistic segmentation masks of input RGB images in a
    pixel-wise manner using the PixelSegmentation module.
    """
    def __init__(self, patch_size: int = 5, nb_channels: int = 3) -> None:
        """Constructor of the class.

        Args:
            patch_size (int, optional): Side length of the square patch. Defaults to 5.
            nb_channels (int, optional): Number of channels in the input tensor.
                Defaults to 3.
        """
        super().__init__()
        
        self._pixel_segmentation = PixelSegmentation(
            patch_size=patch_size,
            nb_channels=nb_channels,
        )
        
        self._patch_size = patch_size
        
        # Set the padding size
        self._padding_size = patch_size // 2
    
    def _unfold(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Extract patches from the RGB images.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, C, H, W).

        Returns:
            torch.Tensor: Batch of image patches (B, HxW, C, patch_size, patch_size).
        """
        # Pad the RGB images to have 1 patch per pixel
        rgb_images = nn.functional.pad(
            rgb_images,
            (self._padding_size,) * 4,
            mode="constant",
            value=0,
        )
        
        # Extract the patches from the RGB images
        patches = rgb_images.unfold(2, self._patch_size, 1)\
            .unfold(3, self._patch_size, 1)
        
        # Permute the dimensions -> (B, H, W, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        
        # Reshape the patches -> (B, HxW, C, patch_size * patch_size)
        patches = patches.contiguous().view(
            1,
            -1,
            3,
            5,
            5,
        )

        return patches
    
    def forward(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images (B, C, H, W).

        Returns:
            torch.Tensor: Predicted segmentation masks (B, H, W).
        """
        # Extract the patches from the RGB images
        patches = self._unfold(rgb_images)
        
        # Predict the probability of each pixel being part of the foreground
        pixel_probabilities = self._pixel_segmentation(patches)
        
        # Reshape the pixel probabilities to form the segmentation masks (B, H, W)
        segmentation_masks = pixel_probabilities.view(
            rgb_images.size(0),
            rgb_images.size(2),
            rgb_images.size(3),
        )
        
        return segmentation_masks


if __name__ == "__main__":
    
    # Test
    patch_size = 5
    nb_channels = 3
    
    # Create an instance of the PixelSegmentation module
    mlp = PixelSegmentation(
        patch_size=patch_size,
        nb_channels=nb_channels,
    )
    
    # Create a random input tensor (B, C, H, W)
    image_patches = torch.rand(4, 100, nb_channels, patch_size, patch_size)
    
    # Pass the input tensor through the network
    output = mlp(image_patches)
    
    # print(torchinfo.summary(mlp, input_size=(1, input_size)))
    
    # Create an instance of the SegmentationMask module
    segmentation_mask = SegmentationMask(
        patch_size=patch_size,
        nb_channels=nb_channels,
    )
    
    # Create a dummy RGB images tensor (B, C, H, W)
    rgb_images = torch.tensor([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]],
        [[9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12]],
        [[13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16]],
        [[17, 17, 17], [18, 18, 18], [19, 19, 19], [20, 20, 20]],
        [[21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24]],
        [[25, 25, 25], [26, 26, 26], [27, 27, 27], [28, 28, 28]],
    ], dtype=torch.float32)
    
    rgb_images = rgb_images.permute(2, 0, 1).unsqueeze(0)
    
    # Pass the RGB images through the network
    seg_mask = segmentation_mask(rgb_images)
