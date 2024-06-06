# Standard libraries
from pathlib import Path
from typing import Optional

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

# Custom modules
from toolbox.evaluation.crop_resize_transform import CropResizeToAspectTransform
from toolbox.evaluation.sequence_segmentation_dataset import SequenceSegmentationData


class BCOT(torch.utils.data.Dataset):
    
    def __init__(
        self,
        root: str,
        scenes_models_dict: Optional[dict] = None,
        frames_per_sequence: int = 5,
        step_between_frames: int = 1,
        resize_transform: Optional[CropResizeToAspectTransform] = None,
    ) -> None:
        
        self._root = Path(root)
        self._scenes_models_dict = scenes_models_dict
        self._frames_per_sequence = frames_per_sequence
        self._step_between_frames = step_between_frames
        self._resize_transform = resize_transform
        
        # Initialize an empty dictionary to store the sequences and their corresponding
        # frames
        self._sequences_frames_idx = {}
        
        # Fill the dictionary with the sequences and their corresponding frames
        self._load_sequences()
        
    def _load_sequences(self):
        
        if self._sequences_frames_idx:
            raise ValueError(
                "The sequences and their corresponding frames have already been loaded."
            )
        
        # Get the paths of the scenes
        scenes = [scene for scene in self._root.iterdir() if scene.is_dir()
                  and scene.name != "models"]
        
        # Filter the scenes if needed
        if self._scenes_models_dict is not None:
            scenes = [
                scene for scene in scenes
                if scene.name in self._scenes_models_dict.keys()
            ]
        
        if not scenes:
            raise ValueError("No scenes found.")
        
        # Go through the scenes
        for scene in scenes:
            
            # Get the paths of the models
            models = [model for model in scene.iterdir() if model.is_dir()]
            
            # Filter the models if needed
            if self._scenes_models_dict is not None:
                models = [
                    model for model in models
                    if model.name in self._scenes_models_dict[scene.name]
                ]
            
            if not models:
                raise ValueError(f"No models found in the scene {scene.name}.")
            
            # Go through the models
            for model in models:
                
                # Get the paths of the frames
                frames = [frame for frame in model.iterdir() if frame.is_file()
                          and frame.suffix == ".png"]
                
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
                    
    def __len__(self):
        
        return len(self._sequences_frames_idx)
        
    def __getitem__(self, idx):
        
        if idx not in self._sequences_frames_idx.keys():
            raise ValueError(f"No sequence found at index {idx}.")
        
        # Load the sequence of frames
        rgbs = torch.stack([
            transforms.ToTensor()(PIL.Image.open(frame))
            for frame in self._sequences_frames_idx[idx]
        ])
        
        # Get the indices of the frames in order to retrieve the associated
        # ground truth poses
        images_idx = [int(frame.stem) for frame in self._sequences_frames_idx[idx]]
        
        # Initialize the ground truth transformation matrices
        TCO = torch.eye(4, dtype=torch.float32).repeat(self._frames_per_sequence, 1, 1)
        
        # Load the ground truth poses
        poses_file = self._sequences_frames_idx[idx][0].parent / "pose.txt"
        
        with open(poses_file, "r") as f:
            lines = f.readlines()
        
        poses = [
            np.array(lines[idx].strip("\t\n").split("\t"),
                     dtype=np.float32).reshape(3, 4) for idx in images_idx
        ]
        poses = np.array(poses)
        
        # Store the ground truth poses
        TCO[:, :3, :] = torch.tensor(poses)
        
        # Retrive the camera intrinsics
        camera_intrinsics_file = self._sequences_frames_idx[idx][0].parents[1] / "K.txt"
        
        with open(camera_intrinsics_file, "r") as f:
            line = f.readline()
        
        K = np.array(
            line[line.find("(")+1:line.find(")")].split(", "),
            dtype=np.float32,
        ).reshape(3, 3)
        K = torch.tensor(K, dtype=torch.float32)
        
        # Get the object label
        object_label = self._sequences_frames_idx[idx][0].parent.name
        
        sample = SequenceSegmentationData(
            rgbs=rgbs,
            TCO=TCO,
            K=K,
            object_label=object_label,
        )
        
        if self._resize_transform is not None:
            sample = self._resize_transform(sample)
        
        return sample


if __name__ == '__main__':
    
    scenes_models_dict = {
        "easy_static_trans": ["Ape", "Cat"],
        "easy_static_suspension": ["Driller", "Tube"],
    }
    
    resize_transform = CropResizeToAspectTransform(
        p=1.0,
        resize=(256, 256),
    )
    
    dataset = BCOT(
        root="data/BCOT",
        scenes_models_dict=scenes_models_dict,
        frames_per_sequence=10,
        step_between_frames=1,
        resize_transform=resize_transform,
    )
    
    print(len(dataset))
    print(dataset[0].rgbs.shape)
    # print(dataset[1])
