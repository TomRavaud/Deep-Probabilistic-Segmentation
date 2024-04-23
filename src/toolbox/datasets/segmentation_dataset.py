from __future__ import annotations

# Standard libraries
from typing import List, Optional, Union, Set, Iterator
import time
from dataclasses import dataclass, replace
import random

# Third-party libraries
import torch
import numpy as np

# Custom modules
from toolbox.datasets.transformations import SceneObservationTransform
from toolbox.datasets.scene_set import (
    IterableSceneSet,
    SceneObservation,
    ObjectData,
)


#TODO: to modify to fit my needs
@dataclass
class SegmentationData:
    """
    Data corresponding to a dataset sample.
    
    rgb: (h, w, 3) uint8
    depth: (bsz, h, w) float32
    bbox: (4, ) int
    K: (3, 3) float32
    TCO: (4, 4) float32
    """
    rgb: np.ndarray
    bbox: np.ndarray
    TCO: np.ndarray
    K: np.ndarray
    depth: Optional[np.ndarray]
    object_data: ObjectData


#TODO: to modify to fit my needs
@dataclass
class BatchSegmentationData:
    """
    A batch of segmentation data.

    rgbs: (bsz, 3, h, w) uint8
    depths: (bsz, h, w) float32
    bboxes: (bsz, 4) int
    TCO: (bsz, 4, 4) float32
    K: (bsz, 3, 3) float32
    """
    rgbs: torch.Tensor
    object_datas: List[ObjectData]
    bboxes: torch.Tensor
    TCO: torch.Tensor
    K: torch.Tensor
    depths: Optional[torch.Tensor] = None

    def pin_memory(self) -> BatchSegmentationData:
        #NOTE: is this function called when using a DataLoader with pin_memory=True?
        print("Pin memory")
        
        self.rgbs = self.rgbs.pin_memory()
        self.bboxes = self.bboxes.pin_memory()
        self.TCO = self.TCO.pin_memory()
        self.K = self.K.pin_memory()
        
        if self.depths is not None:
            self.depths = self.depths.pin_memory()
        
        return self


class ObjectSegmentationDataset(torch.utils.data.IterableDataset):
    """
    Dataset on which a probabilistic segmentation model is trained. The task consists
    in predicting a probabilistic segmentation mask for a given object in an image given
    the image and points sampled along a misaligned 2D contour of the object.
    """
    def __init__(
        self,
        scene_set: IterableSceneSet,
        min_area: Optional[float] = None,
        return_first_object: bool = False,
        keep_labels_set: Optional[Set[str]] = None,
        valid_data_max_attempts: int = 200,
        resize_transform: Optional[SceneObservationTransform] = None,
        rgb_augmentations: Optional[SceneObservationTransform] = None,
        depth_augmentations: Optional[SceneObservationTransform] = None,
        background_augmentations: Optional[SceneObservationTransform] = None,
    ) -> None:
        """Initialize the ObjectSegmentationDataset.

        Args:
            scene_set (IterableSceneSet): Scene set.
            min_area (Optional[float], optional): Minimum area constraint for the object
                to be considered valid in the observation. Defaults to None.
            return_first_object (bool, optional): Whether to return the first valid
                object or a random object in the observation. Defaults to False.
            keep_labels_set (Optional[Set[str]], optional): Set of labels to keep in the
                observation. If None, all labels are kept. Defaults to None.
            valid_data_max_attempts (int, optional): Maximum number of attempts to find
                a valid data in the dataset. Defaults to 200.
            resize_transform (Optional[SceneObservationTransform], optional): Resize
                transformation to apply to the observation. Defaults to None.
            rgb_augmentations (Optional[SceneObservationTransform], optional):
                Augmentations to apply to the RGB image of the observation. Defaults
                to [].
            depth_augmentations (Optional[SceneObservationTransform], optional):
                Augmentations to apply to the depth image of the observation. Defaults
                to [].
            background_augmentations (Optional[SceneObservationTransform], optional):
                Augmentations to apply to the background of the observation. Defaults
                to [].
        """
        self._scene_set = scene_set
        self._min_area = min_area
        
        self._return_first_object = return_first_object
        self._keep_labels_set = keep_labels_set
        self._valid_data_max_attempts = valid_data_max_attempts
        
        # Transformations (resize, augmentations)
        self._resize_transform = resize_transform
        self._rgb_augmentations = rgb_augmentations
        self._depth_augmentations = depth_augmentations
        self._background_augmentations = background_augmentations
        
        # Timings to construct the data from the observation
        self._timings = None
    
    @staticmethod
    def collate_fn(list_data: List[SegmentationData]) -> BatchSegmentationData:
        """Collate a list of SegmentationData into a BatchSegmentationData. It replaces
        the default collate_fn of the DataLoader to handle the custom data type.

        Args:
            list_data (List[SegmentationData]): List of SegmentationData.

        Returns:
            BatchSegmentationData: Batch of SegmentationData.
        """
        batch_data = BatchSegmentationData(
            rgbs=torch.from_numpy(np.stack([d.rgb for d in list_data])).permute(
                0,
                3,
                1,
                2,
            ),
            bboxes=torch.from_numpy(np.stack([d.bbox for d in list_data])),
            K=torch.from_numpy(np.stack([d.K for d in list_data])),
            TCO=torch.from_numpy(np.stack([d.TCO for d in list_data])),
            object_datas=[d.object_data for d in list_data],
        )

        has_depth = [d.depth is not None for d in list_data]
        if all(has_depth):
            batch_data.depths = torch.from_numpy(np.stack([d.depth for d in list_data]))  # type: ignore
            
        return batch_data

    @staticmethod
    def _remove_invisible_objects(obs: SceneObservation) -> SceneObservation:
        """Remove objects that do not appear in the segmentation.
        
        Args:
            obs (SceneObservation): Scene observation.
            
        Returns:
            SceneObservation: Scene observation with only visible objects.
            
        Raises:
            ValueError: If the segmentation is None.
            ValueError: If the object datas are None.
        """
        if obs.segmentation is None:
            raise ValueError("Segmentation is None")
        elif obs.object_datas is None:
            raise ValueError("Object datas are None")
        
        # Get the unique visible ids in the segmentation
        ids_in_segm = np.unique(obs.segmentation)
        ids_visible = set(ids_in_segm[ids_in_segm > 0])
        
        # Get the object datas of the visible objects
        visib_object_datas = [
            object_data
            for object_data in obs.object_datas
            if object_data.unique_id in ids_visible
        ]
        
        # Replace the object datas in the observation
        new_obs = replace(obs, object_datas=visib_object_datas)
        
        return new_obs
    
    def _make_data_from_obs(
        self,
        obs: SceneObservation,
    ) -> Union[SegmentationData, None]:
        """Construct a SegmentationData from a SceneObservation.
        
        A random object in the scene is selected randomly. It is considered valid if:
            1. It is visible enough (its visible 2D area is >= min_area) ;
            2. It belongs to the set of objects to keep (if keep_objects_set isn't
                None).

        Args:
            obs (SceneObservation): Scene observation.

        Returns:
            Union[SegmentationData, None]: Segmentation data or None if no valid object
        """
        obs = ObjectSegmentationDataset._remove_invisible_objects(obs)

        start = time.time()
        timings = {}

        # Apply the transformations (resize, augmentations)
        s = time.time()
        if self._resize_transform is not None:
            obs = self._resize_transform(obs)
        timings["resize_augmentation"] = time.time() - s

        s = time.time()
        if self._background_augmentations is not None:
            obs = self._background_augmentations(obs)
        timings["background_augmentation"] = time.time() - s

        s = time.time()
        if self._rgb_augmentations is not None:
            obs = self._rgb_augmentations(obs)
        timings["rgb_augmentation"] = time.time() - s

        s = time.time()
        if self._depth_augmentations is not None:
            obs = self._depth_augmentations(obs)
        timings["depth_augmentation"] = time.time() - s

        # Get the unique visible ids in the segmentation
        s = time.time()
        unique_ids_visible = set(np.unique(obs.segmentation))
        
        valid_objects = []

        assert obs.object_datas is not None
        assert obs.rgb is not None
        assert obs.camera_data is not None
        
        for obj in obs.object_datas:
            
            assert obj.bbox_modal is not None
            
            valid = False
            
            if obj.unique_id in unique_ids_visible and np.all(obj.bbox_modal) >= 0:
                valid = True

            if valid and self._min_area is not None:
                # We work with the modal bbox, ie the box bounding only the visible
                # pixels of the object
                bbox = obj.bbox_modal
                
                # Area of the bbox
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                
                # Check if the area is greater than the minimum area
                if area >= self._min_area:
                    valid = True
                else:
                    valid = False

            if valid and self._keep_labels_set is not None:
                valid = obj.label in self._keep_labels_set

            if valid:
                valid_objects.append(obj)

        if len(valid_objects) == 0:
            return None

        # Select the first object or a random object
        if self._return_first_object:
            object_data = valid_objects[0]
        else:
            object_data = random.sample(valid_objects, k=1)[0]
        
        assert object_data.bbox_modal is not None

        timings["other"] = time.time() - s
        timings["total"] = time.time() - start

        # Convert timings to milliseconds
        for k, v in timings.items():
            timings[k] = v * 1000

        self._timings = timings

        assert obs.camera_data.K is not None
        assert obs.camera_data.TWC is not None
        assert object_data.TWO is not None
        
        # Add depth to SegmentationData
        data = SegmentationData(
            rgb=obs.rgb,
            depth=obs.depth if obs.depth is not None else None,
            bbox=object_data.bbox_modal,
            K=obs.camera_data.K,
            TCO=(obs.camera_data.TWC.inverse() * object_data.TWO).matrix,
            object_data=object_data,
        )
        
        return data

    def _find_valid_data(
        self,
        iterator: Iterator[SceneObservation],
    ) -> SegmentationData:
        """Find a valid data in the dataset.

        Args:
            iterator (Iterator[SceneObservation]): Iterator over the dataset.

        Raises:
            ValueError: If a valid data cannot be found in the dataset.

        Returns:
            SegmentationData: Segmentation data.
        """
        # Try to find a valid data for a certain number of attempts
        for _ in range(self._valid_data_max_attempts):
            
            # Get the next observation
            obs = next(iterator)
            
            # Construct the SegmentationData from the observation
            data = self._make_data_from_obs(obs)
            
            if data is not None:
                return data
            
        raise ValueError("Cannot find valid image in the dataset")

    def __iter__(self) -> Iterator[SegmentationData]:
        """Iterate over the dataset.

        Yields:
            Iterator[SegmentationData]: Iterator over the dataset.
        """
        # Iterator over the scene dataset
        iterator = iter(self._scene_set)
        
        while True:
            # Find a valid data
            yield self._find_valid_data(iterator)
