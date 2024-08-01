"""
A generic model used to predict lines probabilistic masks from RGB images, binary
masks and RGB images of the lines. The details of the model are abstracted away;
have a look at the probabilistic segmentation models for more information.
"""
# Standard libraries
from typing import Optional

# Third-party libraries
import torch
from torch import nn

# Custom modules
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


class ObjectSegmentationCLinesModel(nn.Module):
    """
    A module that performs object segmentation using a probabilistic segmentation model.
    """
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
    ) -> None:
        """Constructor.

        Args:
            probabilistic_segmentation_model (nn.Module): The probabilistic segmentation
                model. It should take RGB images, binary masks, and RGB images of
                correspondence lines as input, and return the probabilistic masks of
                the correspondence lines as output.
        """
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
            x.clines_rgbs,
        )
        
        return clines_probabilistic_masks


if __name__ == "__main__":
    pass
