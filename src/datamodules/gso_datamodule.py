# Standard libraries
from typing import Any, Dict, Optional

# Third-party libraries
# import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

# Custom modules
from toolbox.datasets.object_set import RigidObjectSet
from toolbox.datasets.segmentation_dataset import ObjectSegmentationDataset
from toolbox.datasets.make_sets import (
    make_object_set,
    make_iterable_scene_set,
)


class GSODataModule(LightningDataModule):
    """`LightningDataModule` for the GSO dataset.

    Google Scanned Objects (GSO) is an open-source collection of over one thousand
    3D-scanned household items released under a Creative Commons license. Authors of
    MegaPose used this set of objets to create a large-scale synthetic dataset for pose
    estimation. It contains 1M images generated using BlenderProc.
    
    This DataModule gathers all the necessary steps to load this GSO-based synthetic
    dataset and prepare it for training, validation, and testing. It also includes the
    transformations to apply to the data.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```
    """
    def __init__(
        self,
        object_set_cfg: Optional[DictConfig] = None,
        scene_set_cfg: Optional[DictConfig] = None,
        dataset_cfg: Optional[DictConfig] = None,
        dataloader_cfg: Optional[DictConfig] = None,
        transformations_cfg: Optional[DictConfig] = None,
    ) -> None:
        """Initialize the GSODataModule.

        Args:
            object_set_cfg (Optional[DictConfig], optional): Configuration for the
                object set. Defaults to None.
            scene_set_cfg (Optional[DictConfig], optional): Configuration for the scene
                set. Defaults to None.
            dataset_cfg (Optional[DictConfig], optional): Configuration for the dataset.
                Defaults to None.
            dataloader_cfg (Optional[DictConfig], optional): Configuration for the
                dataloader. Defaults to None.
            transformations_cfg (Optional[DictConfig], optional): Configuration for the
                transformations to apply to the data. Defaults to None.

        Raises:
            ValueError: If `transformations_cfg` is not a `DictConfig`.
        """
        super().__init__()

        # Allows to access the hyperparameters as `self.hparams` in the LightningModule
        # and store them in the checkpoints.
        self.save_hyperparameters(logger=False)


        # Set transformations
        if transformations_cfg is None:
            self._resize_transform = None
            self._background_augmentations = []
            self._rgb_augmentations = []
            self._depth_augmentations = []
        elif isinstance(transformations_cfg, DictConfig):
            #TODO: here
            self._resize_transform = None
            self._background_augmentations = []
            self._rgb_augmentations = []
            self._depth_augmentations = []
        else:
            raise ValueError("Invalid type for transformations_cfg."
                             "Must be a DictConfig.")
        
        # Resize transform
        # self._resize_transform = CropResizeToAspectTransform(resize=transformations.resize)
        
        # FIXME: do I need to put this in a list?
        # Background augmentations
        # self._background_augmentations = []
        
        # if 'background_augmentations' in transformations:
        #     for aug in transformations.background_augmentations:
        #         aug_type = getattr(toolbox.datasets.augmentations, aug.type)
        #         self._background_augmentations.append(aug_type(**aug.params))
        
        # TODO: do the same for depth and rgb augmentations: with compose
        
        # Background augmentations
        # self._background_augmentations = []
        
        # if transform.background_augmentation:
        #     self._background_augmentations += [
        #         (
        #                 VOCBackgroundAugmentation(transform.background_augmentation.),
        #         ),
        #     ]

        # # Foreground augmentations
        # self._rgb_augmentations = []
        # if apply_rgb_augmentation:
        #     self._rgb_augmentations += [
        #         SceneObsAug(
        #             [
        #                 SceneObsAug(PillowBlur(factor_interval=(1, 3)), p=0.4),
        #                 SceneObsAug(
        #                     PillowSharpness(factor_interval=(0.0, 50.0)),
        #                     p=0.3,
        #                 ),
        #                 SceneObsAug(PillowContrast(factor_interval=(0.2, 50.0)), p=0.3),
        #                 SceneObsAug(
        #                     PillowBrightness(factor_interval=(0.1, 6.0)),
        #                     p=0.5,
        #                 ),
        #                 SceneObsAug(PillowColor(factor_interval=(0.0, 20.0)), p=0.3),
        #             ],
        #             p=0.8,
        #         ),
        #     ]

        # # Depth augmentations
        # self._depth_augmentations = []
        # if apply_depth_augmentation:
        #     # original augmentations
        #     if depth_augmentation_level == 0:
        #         self._depth_augmentations += [
        #             SceneObsAug(DepthBlurTransform(), p=0.3),
        #             SceneObsAug(DepthEllipseDropoutTransform(), p=0.3),
        #             SceneObsAug(DepthGaussianNoiseTransform(std_dev=0.01), p=0.3),
        #             SceneObsAug(DepthMissingTransform(max_missing_fraction=0.2), p=0.3),
        #         ]

        #     # medium augmentation
        #     elif depth_augmentation_level in {1, 2}:
        #         # medium augmentation
        #         self._depth_augmentations += [
        #             SceneObsAug(DepthBlurTransform(), p=0.3),
        #             SceneObsAug(
        #                 DepthCorrelatedGaussianNoiseTransform(
        #                     gp_rescale_factor_min=15.0,
        #                     gp_rescale_factor_max=40.0,
        #                     std_dev=0.01,
        #                 ),
        #                 p=0.3,
        #             ),
        #             SceneObsAug(
        #                 DepthEllipseDropoutTransform(
        #                     ellipse_dropout_mean=175.0,
        #                     ellipse_gamma_shape=5.0,
        #                     ellipse_gamma_scale=2.0,
        #                 ),
        #                 p=0.5,
        #             ),
        #             SceneObsAug(
        #                 DepthEllipseNoiseTransform(
        #                     ellipse_dropout_mean=175.0,
        #                     ellipse_gamma_shape=5.0,
        #                     ellipse_gamma_scale=2.0,
        #                     std_dev=0.01,
        #                 ),
        #                 p=0.5,
        #             ),
        #             SceneObsAug(DepthGaussianNoiseTransform(std_dev=0.01), p=0.1),
        #             SceneObsAug(DepthMissingTransform(max_missing_fraction=0.9), p=0.3),
        #         ]

        #         # Set the depth image to zero occasionally.
        #         if depth_augmentation_level == 2:
        #             self.depth_augmentations.append(
        #                 SceneObsAug(DepthDropoutTransform(), p=0.3),
        #             )
        #             self.depth_augmentations.append(
        #                 SceneObsAug(DepthBackgroundDropoutTransform(), p=0.2),
        #             )
        #         self.depth_augmentations = [
        #             SceneObsAug(self.depth_augmentations, p=0.8),
        #         ]
        #     else:
        #         msg = f"Unknown depth augmentation type {depth_augmentation_level}"
        #         raise ValueError(msg)
        
        
        # Variable to store the object set
        self._object_set: Optional[RigidObjectSet] = None
    
        # Variables to store the datasets
        self._data_train: Optional[Dataset] = None
        self._data_val: Optional[Dataset] = None
        self._data_test: Optional[Dataset] = None
        
        
    def prepare_data(self) -> None:
        """
        Download the dataset. This method is called on 1 GPU/TPU in distributed
        training.
        """
        #TODO: Add code to download the dataset and the models, if not already downloaded.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things
        like random split twice! Also, it is called after `self.prepare_data()` and
        there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or
            `"predict"`. Defaults to ``None``.
        """
        # Create the set of objects
        self._object_set = make_object_set(**self.hparams.object_set_cfg)

        # Load and split datasets only if not loaded already
        if not self._data_train and not self._data_val and not self._data_test:
            
            # Create an iterable scene set from (a/multiple) scene set(s)
            scene_set_train = make_iterable_scene_set(
                **self.hparams.scene_set_cfg,
            )
            
            # Train dataset
            data_train = ObjectSegmentationDataset(
                scene_set_train,
                resize_transform=self._resize_transform,
                background_augmentations=self._background_augmentations,
                rgb_augmentations=self._rgb_augmentations,
                depth_augmentations=self._depth_augmentations,
                **self.hparams.dataset_cfg,
            )
            
            #TODO: How to split the dataset? Separate shards
            self._data_train = data_train
            self._data_val = None
            self._data_test = None

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self._data_train,
            collate_fn=self._data_train.collate_fn,
            # worker_init_fn=worker_init_fn,
            **self.hparams.dataloader_cfg,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        pass

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        pass

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`,
            or `"predict"`. Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the
        datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given
        datamodule `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = GSODataModule() 
    