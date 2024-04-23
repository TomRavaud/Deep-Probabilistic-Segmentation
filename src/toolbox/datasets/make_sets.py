# Standard libraries
from typing import Optional
from pathlib import Path

# Third party libraries
from omegaconf import ListConfig

# Custom modules
from toolbox.datasets.object_set import RigidObjectSet
from toolbox.datasets.gso_object_set import GoogleScannedObjectSet
from toolbox.datasets.scene_set import (
    SceneSet,
    IterableMultiSceneSet,
    IterableSceneSet,
    RandomIterableSceneSet,
)
from toolbox.datasets.web_scene_set import (
    WebSceneSet,
    IterableWebSceneSet,
)

def make_object_set(name: str, dir: str) -> RigidObjectSet:
    """Create a RigidObjectSet object from the a given object set name and location.

    Args:
        name (str): Name of the object set.
        dir (str): Location of the object set directory.

    Raises:
        ValueError: If the object set name is unknown.

    Returns:
        RigidObjectSet: The object set.
    """
    path = Path(dir)
    
    # GSO models
    if name == "gso.orig":
        objset = GoogleScannedObjectSet(path, split="orig")
    elif name == "gso.normalized":
        objset = GoogleScannedObjectSet(path, split="normalized")
    elif name == "gso.panda3d":
        objset = GoogleScannedObjectSet(path, split="panda3d")
    else:
        raise ValueError(f"Unknown object set name: {name}")
    
    return objset


def make_scene_set(
    set_name: str,
    set_path: Path,
    load_depth: bool = False,
    n_frames: Optional[int] = None,
) -> SceneSet:
    """Create a SceneSet object from the a given set name and path.

    Args:
        set_name (str): Name of the set.
        set_path (Path): Path to the scene set directory.
        load_depth (bool, optional): Whether to load depth images or not.
            Defaults to False.
        n_frames (Optional[int], optional): Number of frames to load.
            Defaults to None.

    Raises:
        ValueError: If the set name is unknown.

    Returns:
        SceneSet: The scene set.
    """
    # Datasets in webdataset format
    if set_name.startswith("webdataset."):
        set_name = set_name[len("webdataset.") :]
        scene_set = WebSceneSet(set_path / set_name)

    else:
        raise ValueError(f"Unknown scene set name: {set_name}")

    scene_set.load_depth = load_depth
    
    # Limit the number of frames
    if n_frames is not None:
        assert scene_set.frame_index is not None
        scene_set.frame_index =\
            scene_set.frame_index.iloc[:n_frames].reset_index(drop=True)
    
    return scene_set

def make_iterable_scene_set(
    dir: str,
    sets_cfg: ListConfig,
    input_depth: bool = False,
    sample_buffer_size: int = 1,
    deterministic: bool = False,
) -> IterableMultiSceneSet:
    """Create an iterable set from a list of scene sets configurations.

    Args:
        dir (str): Location of the directory containing the scene sets.
        sets_cfg (ListConfig): Scene sets configurations.
        input_depth (bool, optional): Whether to load depth images or not.
            Defaults to False.
        sample_buffer_size (int, optional): Number of samples to buffer in memory.
            Defaults to 1.
        deterministic (bool, optional): Whether to iterate deterministically or not.
            Defaults to False.

    Returns:
        IterableMultiSceneSet: The iterable set.
    
    Raises:
        ValueError: If the scene set type is unknown.
    """
    path = Path(dir)
    
    # Initialize a list of scene sets
    scene_set_iterators = []
    
    for this_set_config in sets_cfg:
        
        # Create the SceneSet
        scene_set = make_scene_set(
            this_set_config.name,
            load_depth=input_depth,
            set_path=path,
        )
        
        # Convert the SceneSet into an IterableSceneSet
        if isinstance(scene_set, WebSceneSet):
            assert not deterministic
            iterator: IterableSceneSet = IterableWebSceneSet(
                scene_set,
                buffer_size=sample_buffer_size,
            )
        elif isinstance(scene_set, SceneSet):
            iterator = RandomIterableSceneSet(scene_set, deterministic=deterministic)
        else:
            raise ValueError(f"Unknown scene set type: {type(scene_set)}")
        
        # Repeat the set multiple times if needed and add it to the list
        # of scene sets
        for _ in range(this_set_config.n_repeats):
            scene_set_iterators.append(iterator)
    
    # Gather all the scene sets into a single IterableMultiSceneSet
    return IterableMultiSceneSet(scene_set_iterators)
