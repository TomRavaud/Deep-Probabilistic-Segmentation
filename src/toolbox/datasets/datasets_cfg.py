# Standard libraries
from typing import Optional
from pathlib import Path

# Third party libraries
from omegaconf import ListConfig

# Custom modules
from toolbox.datasets.object_set import RigidObjectSet
from toolbox.datasets.gso_object_set import GoogleScannedObjectSet
from toolbox.datasets.scene_dataset import SceneDataset
from toolbox.datasets.web_scene_dataset import (
    WebSceneDataset,
    IterableWebSceneDataset,
)
from toolbox.datasets.scene_dataset import (
    IterableMultiSceneDataset,
    IterableSceneDataset,
    RandomIterableSceneDataset,
)


def make_object_set(objset_name: str, objset_path: Path) -> RigidObjectSet:
    """Create a RigidObjectDataset object from the a given object set name and path.

    Args:
        objset_name (str): Name of the object set.
        objset_path (Path): Path to the object set directory.

    Raises:
        ValueError: If the object set name is unknown.

    Returns:
        RigidObjectDataset: The object set.
    """
    # GSO models
    if objset_name == "gso.orig":
        objset = GoogleScannedObjectSet(objset_path, split="orig")
    elif objset_name == "gso.normalized":
        objset = GoogleScannedObjectSet(objset_path, split="normalized")
    elif objset_name == "gso.panda3d":
        objset = GoogleScannedObjectSet(objset_path, split="panda3d")
    else:
        raise ValueError(f"Unknown dataset name: {objset_name}")
    
    return objset


def make_scene_dataset(
    ds_name: str,
    dataset_path: Path,
    load_depth: bool = False,
    n_frames: Optional[int] = None,
) -> SceneDataset:
    """Create a SceneDataset object from the a given dataset name and path.

    Args:
        ds_name (str): Name of the dataset.
        dataset_path (Path): Path to the dataset directory.
        load_depth (bool, optional): Whether to load depth images or not.
            Defaults to False.
        n_frames (Optional[int], optional): Number of frames to load.
            Defaults to None.

    Raises:
        ValueError: If the dataset name is unknown.

    Returns:
        SceneDataset: The dataset.
    """
    # Datasets in webdataset format
    if ds_name.startswith("webdataset."):
        ds_name = ds_name[len("webdataset.") :]
        ds = WebSceneDataset(dataset_path / ds_name)

    else:
        raise ValueError(ds_name)

    ds.load_depth = load_depth
    if n_frames is not None:
        assert ds.frame_index is not None
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    
    return ds

def make_iterable_scene_dataset(
    dataset_configs: ListConfig,
    dataset_path: Path,
    input_depth: bool = False,
    sample_buffer_size: int = 1,
    deterministic: bool = False,
) -> IterableMultiSceneDataset:
    """Create an iterable dataset from a list of dataset configurations.

    Args:
        dataset_configs (ListConfig): Dataset configurations.
        dataset_path (Path): Path to the dataset directory.
        input_depth (bool, optional): Whether to load depth images or not.
            Defaults to False.
        sample_buffer_size (int, optional): Number of samples to buffer in memory.
            Defaults to 1.
        deterministic (bool, optional): Whether to iterate deterministically or not.
            Defaults to False.

    Returns:
        IterableMultiSceneDataset: The iterable dataset.
    """
    scene_dataset_iterators = []
    
    for this_dataset_config in dataset_configs:
        
        ds = make_scene_dataset(
            this_dataset_config.name,
            load_depth=input_depth,
            dataset_path=dataset_path,
        )
        if isinstance(ds, WebSceneDataset):
            assert not deterministic
            iterator: IterableSceneDataset = IterableWebSceneDataset(
                ds,
                buffer_size=sample_buffer_size,
            )
        else:
            assert isinstance(ds, SceneDataset)
            iterator = RandomIterableSceneDataset(ds, deterministic=deterministic)
        
        for _ in range(this_dataset_config.n_repeats):
            scene_dataset_iterators.append(iterator)
    
    return IterableMultiSceneDataset(scene_dataset_iterators)
