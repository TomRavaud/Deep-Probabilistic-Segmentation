from __future__ import annotations

# Standard libraries
from typing import List, Optional, Union, Set, Iterator, Tuple
import time
from dataclasses import dataclass, replace
import random

# Third-party libraries
import torch
import numpy as np
import cv2

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
    mask: (h, w) uint8
    object_data: ObjectData
    depth: (bsz, h, w) float32
    bbox: (4, ) int
    K: (3, 3) float32
    TCO: (4, 4) float32
    """
    rgb: np.ndarray
    mask: np.ndarray
    bbox: np.ndarray
    TCO: np.ndarray
    DTO: np.ndarray
    K: np.ndarray
    depth: Optional[np.ndarray]
    object_data: ObjectData


#TODO: to modify to fit my needs
@dataclass
class BatchSegmentationData:
    """
    A batch of segmentation data.

    rgbs: (bsz, 3, h, w) uint8
    masks: (bsz, h, w) uint8
    object_datas: List[ObjectData]
    depths: (bsz, h, w) float32
    bboxes: (bsz, 4) int
    TCO: (bsz, 4, 4) float32
    K: (bsz, 3, 3) float32
    """
    rgbs: torch.Tensor
    masks: torch.Tensor
    object_datas: List[ObjectData]
    bboxes: torch.Tensor
    TCO: torch.Tensor
    DTO: torch.Tensor
    K: torch.Tensor
    depths: Optional[torch.Tensor] = None

    def pin_memory(self) -> BatchSegmentationData:
        """Pin memory for the batch.

        Returns:
            BatchSegmentationData: Batch with pinned memory.
        """
        self.rgbs = self.rgbs.pin_memory()
        self.masks = self.masks.pin_memory()
        self.bboxes = self.bboxes.pin_memory()
        self.TCO = self.TCO.pin_memory()
        self.DTO = self.DTO.pin_memory()
        self.K = self.K.pin_memory()
        
        if self.depths is not None:
            self.depths = self.depths.pin_memory()
        
        return self
    
    @property
    def batch_size(self) -> int:
        """Get the batch size.

        Returns:
            int: Batch size.
        """
        return self.rgbs.size(0)

    @property
    def image_size(self) -> Tuple[int, int]:
        """Get the image size.

        Returns:
            Tuple[int, int]: Image size.
        """
        return self.rgbs.size(2), self.rgbs.size(3)


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
        pose_perturbation_prob: float = 0.0,
        rel_translation_scale: Optional[float] = None,
        abs_rotation_scale: Optional[float] = None,
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
        
        self._pose_perturbation_prob = pose_perturbation_prob
        self._rel_translation_scale = rel_translation_scale
        self._abs_rotation_scale = abs_rotation_scale
        
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
            masks=torch.from_numpy(np.stack([d.mask for d in list_data])),
            bboxes=torch.from_numpy(np.stack([d.bbox for d in list_data])),
            K=torch.from_numpy(np.stack([d.K for d in list_data])),
            TCO=torch.from_numpy(np.stack([d.TCO for d in list_data])),
            DTO=torch.from_numpy(np.stack([d.DTO for d in list_data])),
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
    
    @staticmethod
    def _sample_random_pose_perturbation(
        bbox: np.ndarray,
        K: np.ndarray,
        Z: float,
        rel_translation_scale: Optional[float] = None,
        abs_rotation_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Sample a random pose perturbation with respect to the object's pose.

        Args:
            bbox (np.ndarray): Bounding box of the object in the image (modal bbox).
            K (np.ndarray): Camera intrinsic matrix.
            Z (float): Depth of the object in the camera frame.
            translation_scale (float, optional): Relative scale of the translation
                (no units). Defaults to 0.3.
            rotation_scale (float, optional): Absolute scale of the rotation (degrees).
                Defaults to 20.

        Returns:
            np.ndarray: Random pose perturbation represented as a transform matrix.
        """
        DR = np.eye(3)
        Dt = np.zeros(3)
        
        # Get the intrinsic parameters of the camera
        f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
        
        # Compute the 3D coordinates of the bounding box corners in the camera frame
        X1 = Z/f * (bbox[0] - cx)
        X2 = Z/f * (bbox[2] - cx)
        Y1 = Z/f * (bbox[1] - cy)
        Y2 = Z/f * (bbox[3] - cy)
        
        if rel_translation_scale is not None:
            
            # Set the absolute translation scale (adapted to the object size and
            # visibility)
            abs_translation_scale = rel_translation_scale * np.min([X2 - X1, Y2 - Y1])
            
            # Sample a random axis (unit vector)
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            
            # Random translation (uniform distribution in a sphere of radius
            # abs_translation_scale)
            magnitude = np.random.uniform(-abs_translation_scale, abs_translation_scale)
            Dt = magnitude * axis
        
        if abs_rotation_scale is not None:
            
            # Random angle
            angle_deg = random.uniform(-abs_rotation_scale, abs_rotation_scale)
            angle_rad = angle_deg * np.pi / 180
            
            # Random axis (unit vector)
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            
            DR = cv2.Rodrigues(angle_rad * axis)[0]
            
        # Transform matrix
        DT = np.eye(4, dtype=np.float32)
        DT[:3, :3] = DR
        DT[:3, 3] = Dt
        
        return DT
        
    
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
        
        TCO = (obs.camera_data.TWC.inverse() * object_data.TWO).matrix
        
        # Sample a random pose perturbation
        if random.random() <= self._pose_perturbation_prob:
            DTO = ObjectSegmentationDataset._sample_random_pose_perturbation(
                bbox=object_data.bbox_modal,
                K=obs.camera_data.K,
                Z=TCO[2, 3],
                rel_translation_scale=self._rel_translation_scale,
                abs_rotation_scale=self._abs_rotation_scale,
            )
        else:
            DTO = np.eye(4, dtype=np.float32)
        
        # Add depth to SegmentationData
        data = SegmentationData(
            rgb=obs.rgb,
            mask=(obs.segmentation == object_data.unique_id).astype(np.float32),
            depth=obs.depth if obs.depth is not None else None,
            bbox=object_data.bbox_modal,
            K=obs.camera_data.K,
            TCO=TCO,
            DTO=DTO,
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
