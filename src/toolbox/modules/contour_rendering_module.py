# Third-party libraries
import torch
import torch.nn as nn

# Custom modules
from toolbox.datasets.object_set import RigidObjectSet
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


class ContourRendering(nn.Module):
    """
    Module that renders 3D objects, extract their contours and sample points on them.
    """
    def __init__(
        self,
        object_set: RigidObjectSet,
    ) -> None:
        
        super().__init__()
        
        self._object_set = object_set
    
    # TODO: change input to contain only the necessary data and the output to be more
    # explicit
    def forward(self, x: BatchSegmentationData) -> torch.Tensor:
        
        return x
