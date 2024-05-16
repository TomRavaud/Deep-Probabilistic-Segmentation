# Third-party libraries
import torch


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
