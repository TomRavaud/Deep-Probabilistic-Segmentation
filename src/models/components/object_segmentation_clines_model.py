# Standard libraries
from typing import Optional

# Third-party libraries
import torch
from torch import nn

# Custom modules
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


class ObjectSegmentationCLinesModel(nn.Module):
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
    ) -> None:
        super().__init__()
        
        self._probabilistic_segmentation_model = probabilistic_segmentation_model
        
        
    def forward(self, x: BatchSegmentationData) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x (BatchSegmentationData): A batch of segmentation data.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        # Compute the probabilistic segmentation masks
        clines_probabilistic_masks = self._probabilistic_segmentation_model(
            x.rgbs,
            x.masks.unsqueeze(1),
            x.clines_rgb,
            x.clines_binary_masks,
        )
        
        return clines_probabilistic_masks


if __name__ == "__main__":
    pass
