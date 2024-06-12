# Standard libraries
from pathlib import Path
from typing import Optional, List

# TODO: to remove
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import torch
from torchvision import transforms
import PIL
import numpy as np
from omegaconf import DictConfig

# Custom modules
from toolbox.evaluation.crop_resize_transform import CropResizeToAspectTransform
from toolbox.evaluation.sequence_segmentation_dataset import (
    SequenceSegmentationData,
    BatchSequenceSegmentationData,
)


class RBOT(torch.utils.data.Dataset):
    """
    RBOT dataset loading for segmentation evaluation.
    """
    def __init__(
        self,
        root: str,
        scenes_models_dict: Optional[dict] = None,
        frames_per_sequence: int = 5,
        step_between_frames: int = 1,
        transformations_cfg: Optional[DictConfig] = None,
    ) -> None:
        """Constructor.

        Args:
            root (str): Path to the root directory of the dataset.
            scenes_models_dict (Optional[dict], optional): Dictionary of scenes and
                models to load. Defaults to None.
            frames_per_sequence (int, optional): Number of frames per sequence to load.
                Defaults to 5.
            step_between_frames (int, optional): Step between frames to load. Defaults
                to 1.
            transformations_cfg (Optional[DictConfig], optional): Configuration for the
                transformations to apply to the data. Defaults to None.
        """
        self._root = Path(root)
        self._scenes_models_dict = scenes_models_dict
        self._frames_per_sequence = frames_per_sequence
        self._step_between_frames = step_between_frames
        self._resize_transform = None
        
        # Resize transform
        if isinstance(transformations_cfg, DictConfig)\
            and "resize" in transformations_cfg:
            self._resize_transform = CropResizeToAspectTransform(
                resize=transformations_cfg.resize
            )

        # Initialize an empty dictionary to store the sequences and their corresponding
        # frames
        self._sequences_frames_idx = {}
        
        # Fill the dictionary with the sequences and their corresponding frames
        self._load_sequences()
        
    def _load_sequences(self) -> None:
        """Load the sequences of frames.

        Raises:
            ValueError: If the sequences and their corresponding frames have already
                been loaded.
            ValueError: If no scenes are found.
            ValueError: If no models are found in a scene.
        """
        if self._sequences_frames_idx:
            raise ValueError(
                "The sequences and their corresponding frames have already been loaded."
            )
        
        models_scenes_dict = None
        
        if self._scenes_models_dict is not None:
            # Create the set of models
            models = set()
            for models_list in self._scenes_models_dict.values():
                models.update(models_list)

            # Create the reverse dictionary
            models_scenes_dict = {model: [scene for scene, models_list
                                          in self._scenes_models_dict.items()
                                          if model in models_list]
                                  for model in models}
        
        # Get the paths of the models
        models = [model for model in self._root.iterdir() if model.is_dir()]
        
        # Filter the models if needed
        if models_scenes_dict is not None:
            models = [
                model for model in models
                if model.name in models_scenes_dict.keys()
            ]
        
        if not models:
            raise ValueError("No models found.")
        
        # Go through the models
        for model in models:
            
            # Set the scenes associated with the model
            scenes = [
                "a_regular",
                "b_dynamiclight",
                "c_noisy",
                "d_occlusion",
            ]
            
            # Filter the scenes if needed
            if models_scenes_dict is not None:
                scenes = models_scenes_dict[model.name]
            
            # Go through the scenes
            for scene in scenes:
                
                # Get the paths of the frames
                frames_dir = model / "frames/"
                frames = list(frames_dir.glob(f"{scene}*.png"))
                
                if len(frames) < self._frames_per_sequence:
                    continue
                
                frames.sort()
                
                # Extract the sequences of frames
                for i in range(self._step_between_frames):
                    
                    # Extract the frames with a step of self._step_between_frames
                    tmp_frames = frames[i::self._step_between_frames]
                    
                    for j in range(len(tmp_frames) // self._frames_per_sequence):
                            
                        # Extract the sequence of frames
                        sequence = tmp_frames[j * self._frames_per_sequence
                                              :(j + 1) * self._frames_per_sequence]
                        
                        # Store the sequence and its corresponding frames
                        self._sequences_frames_idx[len(self._sequences_frames_idx)] =\
                            sequence
                    
    def __len__(self) -> int:
        """Get the number of sequences.

        Returns:
            int: Number of sequences.
        """
        return len(self._sequences_frames_idx)
        
    def __getitem__(self, idx: int) -> SequenceSegmentationData:
        """Get the sequence of frames at the given index.

        Args:
            idx (int): Index of the sequence.

        Raises:
            ValueError: If no sequence is found at the given index.

        Returns:
            SequenceSegmentationData: Sequence of segmentation data.
        """
        if idx not in self._sequences_frames_idx.keys():
            raise ValueError(f"No sequence found at index {idx}.")
        
        # Load the sequence of frames
        rgbs = torch.stack([
            transforms.ToTensor()(PIL.Image.open(frame))
            for frame in self._sequences_frames_idx[idx]
        ])
        
        # Convert images from float32 to uint8
        rgbs = (rgbs * 255).to(torch.uint8)
        
        # Get the indices of the frames in order to retrieve the associated
        # ground truth poses
        images_idx = [int(frame.stem[-4:]) for frame in self._sequences_frames_idx[idx]]
        
        # Initialize the ground truth transformation matrices
        TCO = torch.eye(4, dtype=torch.float32).repeat(self._frames_per_sequence, 1, 1)
        
        # Load the ground truth poses
        poses_file = self._sequences_frames_idx[idx][0].parents[2] / "poses_first.txt"
        
        with open(poses_file, "r") as f:
            f.readline()  # Skip the first line
            lines = f.readlines()
        
        poses = np.array([
            np.array(lines[idx].strip("\n").split("\t"),
                     dtype=np.float32) for idx in images_idx
        ])
        
        # Store the ground truth poses
        TCO[:, :3, :3] = torch.tensor(poses[:, :9].reshape(-1, 3, 3))  # Rotation
        TCO[:, :3, 3] = torch.tensor(poses[:, 9:].reshape(-1, 3))  # Translation
        
        
        # Initialize the ground truth transformation matrices for the occluding object
        TCO_occluder =\
            torch.eye(4, dtype=torch.float32).repeat(self._frames_per_sequence, 1, 1)
        
        # Load the poses of the occluding object
        poses_occluder_file =\
            self._sequences_frames_idx[idx][0].parents[2] / "poses_second.txt"
        
        with open(poses_occluder_file, "r") as f:
            f.readline()  # Skip the first line
            lines = f.readlines()
        
        poses_occluder = np.array([
            np.array(lines[idx].strip("\n").split("\t"),
                     dtype=np.float32) for idx in images_idx
        ])
        
        # Store the ground truth poses
        TCO_occluder[:, :3, :3] =\
            torch.tensor(poses_occluder[:, :9].reshape(-1, 3, 3))  # Rotation
        TCO_occluder[:, :3, 3] =\
            torch.tensor(poses_occluder[:, 9:].reshape(-1, 3))  # Translation

        # Retrive the camera intrinsics
        camera_intrinsics_file =\
            self._sequences_frames_idx[idx][0].parents[2] / "camera_calibration.txt"
        
        with open(camera_intrinsics_file, "r") as f:
            f.readline()  # Skip the first line
            line = f.readline()
    
        fx, fy, cx, cy = line.split("\t")[:4]
        
        K = np.array([
            [float(fx), 0, float(cx)],
            [0, float(fy), float(cy)],
            [0, 0, 1],
        ])
        K = torch.tensor(K, dtype=torch.float32)
        
        # Get the object label
        object_label = self._sequences_frames_idx[idx][0].parents[1].name
        
        sample = SequenceSegmentationData(
            rgbs=rgbs,
            TCO=TCO,
            K=K,
            object_label=object_label,
        )
        
        if self._resize_transform is not None:
            sample = self._resize_transform(sample)
        
        return sample
    
    @staticmethod
    def collate_fn(
        list_data: List[SequenceSegmentationData],
    ) -> BatchSequenceSegmentationData:
        """Collate a list of SegmentationData into a BatchSegmentationData. It replaces
        the default collate_fn of the DataLoader to handle the custom data type.

        Args:
            list_data (List[SegmentationData]): List of SegmentationData.

        Returns:
            BatchSegmentationData: Batch of SegmentationData.
        """
        batch_data = BatchSequenceSegmentationData(
            rgbs=torch.stack([d.rgbs for d in list_data]),
            K=torch.stack([d.K for d in list_data]),
            TCO=torch.stack([d.TCO for d in list_data]),
            object_labels=[d.object_label for d in list_data],
        )

        return batch_data


if __name__ == '__main__':
    
    import cv2
    
    scenes_models_dict = {
        "a_regular": ["ape", "cat"],
        "b_dynamiclight": ["ape", "phone"],
    }
    
    transformations_cfg = DictConfig(
        {
            "resize": [240, 320],
        }
    )
    
    dataset = RBOT(
        root="data/RBOT",
        scenes_models_dict=scenes_models_dict,
        frames_per_sequence=10,
        step_between_frames=3,
        transformations_cfg=transformations_cfg,
    )
    
    print(len(dataset))
    print(dataset[0].rgbs.shape)
     
    img = dataset[0].rgbs[0]
    img = img.permute(1, 2, 0).numpy()
    
    # Image to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
