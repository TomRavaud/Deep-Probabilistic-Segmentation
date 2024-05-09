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

    # def __getitem__(self, idx: int) -> SceneObservation:
    #     assert self.frame_index is not None
    #     row = self.frame_index.iloc[idx]
    #     shard_id, key = row.shard_id, row.key
    #     shard_path = self.wds_dir / f"shard-{shard_id:06d}.tar"

    #     bop_obs = bop_webdataset.load_image_data(
    #         shard_path,
    #         key,
    #         load_rgb=True,
    #         load_mask_visib=True,
    #         load_gt=True,
    #         load_gt_info=True,
    #     )
    #     obs = data_from_bop_obs(bop_obs, use_raw_object_id=True)
    #     return obs

def load_scene_ds_obs(
    sample: Dict[str, Union[bytes, str]],
    depth_scale: float = 1000.0,
    load_depth: bool = False,
    label_format: str = "{label}",
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
