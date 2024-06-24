from __future__ import annotations

# Standard libraries
from typing import List, Tuple
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
    # Name of the scene
    scene_label: str

@dataclass
class BatchSequenceSegmentationData:
    """
    Batch of sequence segmentation data.
    """
    # Batch of sequences of RGB images
    # Shape: (B, T, C, H, W), B the batch size, T the number of images in the sequence
    # Range: [0, 255]
    # Data type: torch.uint8
    rgbs: torch.Tensor
    
    # Batch of ground truth transformation matrices from object to camera
    # Shape: (B, T, 4, 4)
    # Data type: torch.float32
    TCO: torch.Tensor
    
    # Batch of camera intrinsics
    # Shape: (B, 3, 3)
    # Data type: torch.float32
    K: torch.Tensor
    
    # Batch of object labels
    object_labels: List[str]
    # Batch of scene labels
    scene_labels: List[str]

    def pin_memory(self) -> BatchSequenceSegmentationData:
        """Pin memory for the batch.

        Returns:
            BatchSequenceSegmentationData: Batch with pinned memory.
        """
        self.rgbs = self.rgbs.pin_memory()
        self.TCO = self.TCO.pin_memory()
        self.K = self.K.pin_memory()
        
        return self
    
    @property
    def batch_size(self) -> int:
        """Get the batch size.

        Returns:
            int: Batch size.
        """
        return self.rgbs.size(0)
    
    @property
    def sequence_size(self) -> int:
        """Get the sequence size.

        Returns:
            int: Sequence size.
        """
        return self.rgbs.size(1)
    
    @property
    def image_size(self) -> Tuple[int, int]:
        """Get the image size.

        Returns:
            Tuple[int, int]: Image size.
        """
        return self.rgbs.size(3), self.rgbs.size(4)
    
    def to(self, *args, **kwargs) -> BatchSequenceSegmentationData:
        """Move the batch to a device.

        Returns:
            BatchSequenceSegmentationData: Batch moved to a device.
        """
        self.rgbs = self.rgbs.to(*args, **kwargs)
        self.TCO = self.TCO.to(*args, **kwargs)
        self.K = self.K.to(*args, **kwargs)
        
        return self
    

# TODO: to keep or remove?
class SequenceSegmentationDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

