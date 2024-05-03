# Third-party libraries
import torch
import torch.nn as nn


class SegmentationMask(nn.Module):
    """
    Module that computes a segmentation mask from a probabilistic segmentation model.
    """
    def __init__(self):
        super().__init__()
    
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
    
    def forward(
        self,
        rgb_images: torch.Tensor,
        implicit_segmentations: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the module.

        Args:
            rgb_images (torch.Tensor): Batch of RGB images.
            implicit_segmentations (torch.Tensor): Batch of implicit segmentations
                vectors.

        Returns:
            torch.Tensor: Batch of segmentation masks.
        """
        # Convert RGB images to HSV
        hsv_images = self.rgb2hsv_torch(rgb_images)
        
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
        segmentation_masks = torch.gather(
            implicit_segmentations,
            1,
            scaled_hue,
        )
        
        return segmentation_masks


if __name__ == "__main__":
    pass
