# Third party libraries
import torch

# Custom modules
from toolbox.utils.rgb2hsv_torch import rgb2hsv_torch


class SegmentationWithHistograms():
    """
    Module that computes the probability of each pixel in an image being part of the
    foreground based on the hue values of the pixels. To do so, it computes histograms
    of the hue values for the foreground and the background of each image in a batch,
    and applies the Bayes' rule to get the probabilities given the hue values.
    """
    def __init__(
        self,
        nb_bins: tuple = (10, 10, 10),
        color_space: str = "rgb",
    ) -> None:
        
        self._nb_bins = nb_bins
        self._color_space = color_space
    
    def _forward_hue(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            x (torch.Tensor): Batch of RGB images and binary masks (B, 4, H, W).
                RGB values are in the range [0, 1] and the binary masks values are 0
                or 1.
            
        Returns:
            torch.Tensor: Probability of each pixel in the image being part of the
                foreground (B, output_dim).
        """
        # Extract the batched RGB images and masks
        rgb_images = x[:, :3]
        binary_masks = x[:, 3]
        
        # Convert the RGB images to HSV
        hsv_images = rgb2hsv_torch(rgb_images)
        
        # Extract the hue channel
        hue_channel = hsv_images[:, 0, :, :]  # assuming hue is the first channel
        
        # Scale the hue values to the range [0, output_dim] and convert them to integers
        scaled_hue = (hue_channel * self._nb_bins[0]).long()
        
        # Compute a foreground histogram and a background histogram for each image of
        # the batch
        foreground_histograms = torch.zeros(
            (x.size(0), self._nb_bins[0]),
            device=x.device,
        )
        background_histograms = torch.zeros(
            (x.size(0), self._nb_bins[0]),
            device=x.device,
        )
        for i in range(x.size(0)):
            # Extract the hue values of the foreground pixels
            foreground_hue_values = scaled_hue[i][binary_masks[i] == 1]
            
            # Extract the hue values of the background pixels
            background_hue_values = scaled_hue[i][binary_masks[i] == 0]
            
            # Compute the histograms
            foreground_histograms[i] = torch.bincount(
                foreground_hue_values,
                minlength=self._nb_bins[0],
            )
            background_histograms[i] = torch.bincount(
                background_hue_values,
                minlength=self._nb_bins[0],
            )
        
        # Compute the area of the foreground and the background regions
        total_pixels = scaled_hue.size(1) * scaled_hue.size(2)
        foreground_pixels = binary_masks.sum(dim=(1, 2))
        background_pixels = total_pixels - foreground_pixels
        foreground_area = foreground_pixels / total_pixels
        background_area = background_pixels / total_pixels
        
        # Transform the histograms counts into frequencies
        foreground_histograms /= foreground_pixels.view(-1, 1)
        background_histograms /= background_pixels.view(-1, 1)
        
        # Apply Bayes' rule to get the probabilities given the hue values
        p_foreground = (foreground_histograms * foreground_area.view(-1, 1)) / (
            foreground_histograms * foreground_area.view(-1, 1) +
            background_histograms * background_area.view(-1, 1)
        )
        
        # Replace NaN values with 0
        p_foreground[p_foreground != p_foreground] = 0.0
        
        return p_foreground
    
    def _forward_rgb(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            x (torch.Tensor): Batch of RGB images and binary masks (B, 4, H, W).
                RGB values are in the range [0, 1] and the binary masks values are 0
                or 1.
            
        Returns:
            torch.Tensor: Probability of each pixel in the image being part of the
                foreground (B, output_dim).
        """
        # Extract the batched RGB images and masks
        rgb_images = x[:, :3]
        binary_masks = x[:, 3]
        
        # Compute a foreground histogram and a background histogram for each image of
        # the batch
        foreground_histograms = torch.zeros(
            (x.size(0), *self._nb_bins),
            device=x.device,
        )
        background_histograms = torch.zeros(
            (x.size(0), *self._nb_bins),
            device=x.device,
        )
        
        for i in range(x.size(0)):
            # Extract the color values of the foreground pixels
            # (ensure that the values are on the CPU for implementation reasons)
            foreground_rgb_values = rgb_images[i][:, binary_masks[i] == 1].cpu()
            
            # Extract the color values of the background pixels
            # (ensure that the values are on the CPU for implementation reasons)
            background_rgb_values = rgb_images[i][:, binary_masks[i] == 0].cpu()
            
            # Compute the histograms
            foreground_histograms[i], _ = torch.histogramdd(
                foreground_rgb_values.permute(1, 0),
                bins=self._nb_bins,
                range=[0., 1.]*len(self._nb_bins),
            )
            background_histograms[i], _ = torch.histogramdd(
                background_rgb_values.permute(1, 0),
                bins=self._nb_bins,
                range=[0., 1.]*len(self._nb_bins),
                density=True,
            )
        
        # Compute the area of the foreground and the background regions
        # total_pixels = scaled_hue.size(1) * scaled_hue.size(2)
        total_pixels = binary_masks.size(1) * binary_masks.size(2)
        foreground_pixels = binary_masks.sum(dim=(1, 2))
        background_pixels = total_pixels - foreground_pixels
        foreground_area = foreground_pixels / total_pixels
        background_area = background_pixels / total_pixels
        
        # Add dimensions to perform broadcasting
        new_dims = (None,) * len(self._nb_bins)
        foreground_area = foreground_area[(slice(None), *new_dims)]
        background_area = background_area[(slice(None), *new_dims)]
        
        # Apply Bayes' rule to get the probabilities given the hue values
        p_foreground = (foreground_histograms * foreground_area) / (
            foreground_histograms * foreground_area +
            background_histograms * background_area
        )
        
        # Replace NaN values with 0
        p_foreground[p_foreground != p_foreground] = 0.0
        
        return p_foreground
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the forward pass of the module.

        Args:
            x (torch.Tensor): Batch of RGB images and binary masks (B, 4, H, W).
                RGB values are in the range [0, 1] and the binary masks values are 0
                or 1.

        Returns:
            torch.Tensor: Probability of each pixel in the image being part of the
                foreground (B, output_dim).
        """
        if self._color_space == "h":
            return self._forward_hue(x)
        elif self._color_space == "rgb":
            return self._forward_rgb(x)
        else:
            raise ValueError(f"Invalid color space: {self._color_space}.")


if __name__ == "__main__":
    
    # For debugging purposes only
    import cv2
    import matplotlib.pyplot as plt
    
    # RGB image
    image = cv2.imread("notebooks/images/cat_rbot.png")
    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Mask
    mask = cv2.imread("notebooks/images/cat_rbot_mask.png", cv2.IMREAD_GRAYSCALE)
    
    # 0-255 -> 0-1
    mask = mask / 255.0
    
    ## Test the SegmentationWithHistograms module
    module = SegmentationWithHistograms(color_space="rgb")
    
    input1 = torch.cat([
        torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0,
        torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
    ], dim=0)
    
    input2 = torch.cat([
        torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0,
        torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
    ], dim=0)

    # Batch input
    input = torch.stack([input1, input2], dim=0)
    
    p_foreground = module(input)
    
    print(p_foreground.shape)
