# Standard libraries
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Dict, Union
import json

# Third party libraries
import pandas as pd
import webdataset as wds
import numpy as np
import imageio
import io
import pyarrow.feather as feather

# Custom modules
from toolbox.utils.webdataset import tarfile_to_samples
from toolbox.datasets.scene_set import (
    IterableSceneSet,
    SceneSet,
    SceneObservation,
    CameraData,
    ObservationInfos,
    ObjectData,
    DataJsonType,
)


class WebSceneSet(SceneSet):
    def __init__(
        self,
        wds_dir: Path,
        split_range: tuple = (0., 1.),
        load_depth: bool = True,
        load_segmentation: bool = True,
        label_format: str = "{label}",
        load_frame_index: bool = False,
    ):
        self.label_format = label_format
        self.wds_dir = wds_dir
        self.split_range = split_range

        frame_index = None
        if load_frame_index:
            key_to_shard = json.loads((wds_dir / "key_to_shard.json").read_text())
            frame_index = defaultdict(list)
            for key, shard_id in key_to_shard.items():
                image_id, scene_id = map(int, key.split("_"))
                frame_index["image_id"].append(image_id)
                frame_index["scene_id"].append(scene_id)
                frame_index["key"].append(key)
                frame_index["shard_id"].append(shard_id)
            frame_index = pd.DataFrame(frame_index)

        super().__init__(
            frame_index=frame_index,
            load_depth=load_depth,
            load_segmentation=load_segmentation,
        )
        
        
        #TODO: to clean
        # Path to the directory used to map scene_is and view_id to shard_id and key
        self._index_frame_dir = Path("data/webdatasets/frame_index")
        
        # Check if the index frame directory exists
        if not self.index_frame_dir.exists():
            raise ValueError(
                f"Index frame directory does not exist: {self.index_frame_dir}"
            )
    
    @property
    def index_frame_dir(self) -> Path:
        """Get the index frame directory.
        
        Returns:
            Path: The index frame directory.
        """
        return self._index_frame_dir

    def get_tar_list(self) -> List[str]:
        """Get the list of tar files in the dataset directory.

        Returns:
            List[str]: List of tar files.
        """
        tar_files = [str(x) for x in self.wds_dir.iterdir() if x.suffix == ".tar"]
        tar_files.sort()
        
        if self.split_range[0] < 0. or\
            self.split_range[1] > 1. or\
                self.split_range[0] >= self.split_range[1]:
            raise ValueError(f"Invalid split range: {self.split_range}")
        
        tar_files = tar_files[
            int(len(tar_files) * self.split_range[0]):
                int(len(tar_files) * self.split_range[1])]
        
        return tar_files


def load_scene_ds_obs(
    sample: Dict[str, Union[bytes, str]],
    depth_scale: float = 1000.0,
    load_depth: bool = False,
    label_format: str = "{label}",
    index_frame_dir: Path = None,
) -> SceneObservation:
    
    assert isinstance(sample["rgb.png"], bytes)
    assert isinstance(sample["segmentation.png"], bytes)
    assert isinstance(sample["depth.png"], bytes)
    assert isinstance(sample["camera_data.json"], bytes)
    assert isinstance(sample["infos.json"], bytes)

    rgb = np.array(imageio.imread(io.BytesIO(sample["rgb.png"])))
    segmentation = np.array(imageio.imread(io.BytesIO(sample["segmentation.png"])))
    segmentation = np.asarray(segmentation, dtype=np.uint32)
    depth = None
    if load_depth:
        depth = imageio.imread(io.BytesIO(sample["depth.png"]))
        depth = np.asarray(depth, dtype=np.float32)
        depth /= depth_scale

    object_datas_json: List[DataJsonType] = json.loads(sample["object_datas.json"])
    object_datas = [ObjectData.from_json(d) for d in object_datas_json]
    
    for obj in object_datas:
        obj.label = label_format.format(label=obj.label)

    camera_data = CameraData.from_json(sample["camera_data.json"])
    infos = ObservationInfos.from_json(sample["infos.json"])
    
    
    if index_frame_dir is not None:
        scene_id = infos.scene_id
        view_id = infos.view_id
        
        # Read the corresponding row from the index dataframe
        df = feather.read_feather(index_frame_dir / f"{scene_id}.feather")
        row = df[df.view_id == view_id]

        # Extract the key and the shard_id
        key = row["key"].values[0]
        shard_id = row["shard_id"].values[0]

        # Add the key and the shard_id to the infos
        infos.key = key
        infos.shard_id = shard_id
    
    return SceneObservation(
        rgb=rgb,
        depth=depth,
        segmentation=segmentation,
        infos=infos,
        object_datas=object_datas,
        camera_data=camera_data,
    )

class IterableWebSceneSet(IterableSceneSet):
    """
    Iterable scene set for webdataset format.
    """
    def __init__(self, web_scene_set: WebSceneSet, buffer_size: int = 1) -> None:
        """Constructor.

        Args:
            web_scene_set (WebSceneSet): The web scene set.
            buffer_size (int, optional): Number of samples to buffer in memory
                before shuffling. Defaults to 1.

        Yields:
            Iterator: Iterator over SceneObservation objects.
        """
        self.web_scene_set = web_scene_set
        
        load_scene_ds_obs_ = partial(
            load_scene_ds_obs,
            # depth_scale=self.web_scene_set.depth_scale,
            load_depth=self.web_scene_set.load_depth,
            label_format=self.web_scene_set.label_format,
            index_frame_dir=self.web_scene_set.index_frame_dir,
        )

        def load_scene_ds_obs_iterator(
            samples,
        ):
            for sample in samples:
                yield load_scene_ds_obs_(sample)

        # Create the webdataset Pipeline (wds.Dataset is a shorthand for writing down
        # pipelines, but the underlying pipeline is an instance of wds.DataPipeline)
        self.datapipeline = wds.DataPipeline(
            # Sample from the shards
            wds.ResampledShards(self.web_scene_set.get_tar_list()),
            
            # FIXME: Should we split by worker here? cf webdataset github
            # wds.split_by_worker,
            
            # Extract samples from the tar file
            tarfile_to_samples(),
            
            # Convert a sample to a SceneObservation
            load_scene_ds_obs_iterator,
            
            # Shuffle the samples
            wds.shuffle(buffer_size),
        )

    def __iter__(self):
        return iter(self.datapipeline)
