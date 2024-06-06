# Standard libraries
from typing import Optional
from dataclasses import dataclass

# Third-party libraries
import torch


@dataclass
class SequenceSegmentationData:
    """
    Data corresponding to a sequence of images on which segmentation evaluation
    is performed
    """
    # Sequence of RGB images
    # Shape: (T, C, H, W), T the number of images in the sequence
    # Range: [0, 255]
    # Data type: torch.uint8
    rgbs: torch.Tensor
    
    # Ground truth transformation matrices from object to camera
    # Shape: (T, 4, 4)
    # Data type: torch.float32
    TCO: torch.Tensor
    
    # Camera intrinsics
    # Shape: (3, 3)
    # Data type: torch.float32
    K: torch.Tensor
    
    # Name of the object
    object_label: str
    

# TODO: to keep or remove?
class SequenceSegmentationDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

