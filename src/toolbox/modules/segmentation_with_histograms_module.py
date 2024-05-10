# Third party libraries
import torch


class SegmentationWithHistograms():
    """
    Module that computes the probability of each pixel in an image being part of the
    foreground based on the hue values of the pixels. To do so, it computes histograms
    of the hue values for the foreground and the background of each image in a batch,
    and applies the Bayes' rule to get the probabilities given the hue values.
    """
    def __init__(
        self,
        output_dim: int = 180,
    ) -> None:
        # Number of bins in the histograms (the bin size is 1)
        self._output_dim = output_dim
    
    @staticmethod
    def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB images to HSV images.
        
        Source:
        https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py#L28C1-L41C51
        
        Args:
            rgb (torch.Tensor): Batch of RGB images.

        Returns:
            torch.Tensor: Batch of HSV images.
        """
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        
        # To ensure the hue is in the range [0, 1] (multiply by 360 to get the hue in
        # degrees)
        hsv_h /= 6.
        
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v = cmax
        
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)
    
    def _forward(
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
        hsv_images = self.rgb2hsv_torch(rgb_images)
        
        # Extract the hue channel
        hue_channel = hsv_images[:, 0, :, :]  # assuming hue is the first channel
        
        # Scale the hue values to the range [0, output_dim] and convert them to integers
        scaled_hue = (hue_channel * self._output_dim).long()
        
        # Compute a foreground histogram and a background histogram for each image of
        # the batch
        foreground_histograms = torch.zeros(
            (x.size(0), self._output_dim),
            device=x.device,
        )
        background_histograms = torch.zeros(
            (x.size(0), self._output_dim),
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
                minlength=self._output_dim,
            )
            background_histograms[i] = torch.bincount(
                background_hue_values,
                minlength=self._output_dim,
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
        return self._forward(x)


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
    module = SegmentationWithHistograms()
    
    input = torch.cat([
        torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0,
        torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    ], dim=1)
    
    p_foreground = module(input)
    
    plt.figure()
    plt.plot(p_foreground[0].cpu().numpy())
    plt.show()
