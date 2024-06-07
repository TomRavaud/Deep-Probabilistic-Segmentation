# Standard libraries
import random
from copy import deepcopy

# TODO: to remove
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import numpy as np
import torch
from torchvision.transforms.functional import crop, resize

# Custom modules
from toolbox.geometry.camera_geometry import get_K_crop_resize
from toolbox.evaluation.sequence_segmentation_dataset import SequenceSegmentationData


class CropResizeToAspectTransform:
    """
    Crop and resize the RGB observations to a target aspect ratio.
    """
    def __init__(self, resize: tuple = (480, 640), p: float = 1.0) -> None:
        """Constructor.

        Args:
            resize (Resolution, optional): Target aspect ratio (height, width).
                Defaults to (480, 640).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.

        Raises:
            ValueError: If the width is less than the height.
        """
        self._p = p
        
        if resize[1] < resize[0]:
            raise ValueError("The width must be greater than the height.")
        
        self._resize = resize
        self._aspect = max(resize) / min(resize)
    
    def _transform(self, seq: SequenceSegmentationData) -> SequenceSegmentationData:
        """Crop and resize the RGB observations to a target aspect ratio.

        Args:
            seq (SequenceSegmentationData): Sequence of segmentation data.

        Raises:
            ValueError: If the sequence of RGB images is None.
            ValueError: If the camera intrinsics are None.

        Returns:
            SequenceSegmentationData: Transformed sequence of segmentation data.
        """
        if seq.rgbs is None:
            raise ValueError("The sequence of RGB images is None.")
        elif seq.K is None:
            raise ValueError("The camera intrinsics are None.")
        
        h, w = seq.rgbs.shape[2:4]

        # Skip if the image is already at the target size
        if (h, w) == self._resize:
            return seq

        # Match the width on input image with an image of target aspect ratio.
        if not np.isclose(w / h, self._aspect):
            r = self._aspect
            crop_h = w * 1 / r
            x0, y0 = w / 2, h / 2
            crop_box_size = (crop_h, w)
            crop_h, crop_w = min(crop_box_size), max(crop_box_size)
            x1, y1, x2, y2 = (
                x0 - crop_w / 2,
                y0 - crop_h / 2,
                x0 + crop_w / 2,
                y0 + crop_h / 2,
            )
            box = (x1, y1, x2, y2)
            box = [int(b) for b in box]
            
            # Crop the RGB images
            rgbs = crop(seq.rgbs, box[1], box[0], box[3] - box[1], box[2] - box[0])
            
            new_K = get_K_crop_resize(
                seq.K.unsqueeze(0),
                torch.tensor(box).unsqueeze(0),
                orig_size=(h, w),
                crop_resize=(crop_h, crop_w),
            )[0]
            
        else:
            new_K = seq.K

        # Resize to target size
        h, w = rgbs.shape[2:4]
        
        w_resize, h_resize = max(self._resize), min(self._resize)
        rgbs = resize(rgbs, (h_resize, w_resize), antialias=True)
        
        box = (0, 0, w, h)
        new_K = get_K_crop_resize(
            new_K.unsqueeze(0),
            torch.tensor(box).unsqueeze(0),
            orig_size=(h, w),
            crop_resize=(h_resize, w_resize),
        )[0]

        new_seq = deepcopy(seq)
        new_seq.K = new_K
        new_seq.rgbs = rgbs
        
        return new_seq
    
    def __call__(self, seq: SequenceSegmentationData) -> SequenceSegmentationData:
        """Apply or not the transformation to the observation given the
        probability `p`.

        Args:
            seq (SequenceSegmentationData): Sequence observation.

        Returns:
            SequenceSegmentationData: Transformed sequence observation.
        """
        if random.random() <= self._p:
            return self._transform(seq)
        else:
            return seq


if __name__ == "__main__":
    
    input_data = SequenceSegmentationData(
        rgbs=torch.randint(0, 255, (10, 3, 512, 640), dtype=torch.uint8),
        TCO=torch.eye(4, dtype=torch.float32).repeat(10, 1, 1),
        K=torch.eye(3, dtype=torch.float32),
        object_label="object",
    )
    
    transform = CropResizeToAspectTransform()
    
    transformed_data = transform(input_data)
    
    print(transformed_data.rgbs.shape)
    print(transformed_data.K.shape)
