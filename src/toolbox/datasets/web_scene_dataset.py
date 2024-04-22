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
from toolbox.datasets.scene_dataset import (
    IterableSceneDataset,
    SceneDataset,
    SceneObservation,
    CameraData,
    ObservationInfos,
    ObjectData,
    DataJsonType,
)


class WebSceneDataset(SceneDataset):
    def __init__(
        self,
        wds_dir: Path,
        load_depth: bool = True,
        load_segmentation: bool = True,
        label_format: str = "{label}",
        load_frame_index: bool = False,
    ):
        self.label_format = label_format
        self.wds_dir = wds_dir

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
        tar_files = [str(x) for x in self.wds_dir.iterdir() if x.suffix == ".tar"]
        tar_files.sort()
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

class IterableWebSceneDataset(IterableSceneDataset):
    def __init__(self, web_scene_dataset: WebSceneDataset, buffer_size: int = 1):
        self.web_scene_dataset = web_scene_dataset

        load_scene_ds_obs_ = partial(
            load_scene_ds_obs,
            # depth_scale=self.web_scene_dataset.depth_scale,
            load_depth=self.web_scene_dataset.load_depth,
            label_format=self.web_scene_dataset.label_format,
        )

        def load_scene_ds_obs_iterator(
            samples,
        ):
            for sample in samples:
                yield load_scene_ds_obs_(sample)

        self.datapipeline = wds.DataPipeline(
            wds.ResampledShards(self.web_scene_dataset.get_tar_list()),
            tarfile_to_samples(),
            load_scene_ds_obs_iterator,
            wds.shuffle(buffer_size),
        )

    def __iter__(self):
        return iter(self.datapipeline)
