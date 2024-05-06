# Standard libraries
from typing import Any, Dict, Optional

# Third-party libraries
# import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, ListConfig

# Custom modules
from toolbox.datasets.segmentation_dataset import ObjectSegmentationDataset
from toolbox.datasets.make_sets import make_iterable_scene_set
import toolbox.datasets.transformations as transformations


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
        """
        super().__init__()

        # Allows to access the hyperparameters as `self.hparams` in the LightningModule
        # and store them in the checkpoints.
        self.save_hyperparameters(logger=False)

        # Set transformations
        self._resize_transform = None
        self._background_augmentations = None
        self._rgb_augmentations = None
        self._depth_augmentations = None
        
        if isinstance(transformations_cfg, DictConfig):
            
            # Resize transform
            if "resize" in transformations_cfg:
                self._resize_transform = transformations.CropResizeToAspectTransform(
                    resize=transformations_cfg.resize
                )
            
            # RGB augmentations
            if "rgb_augmentations" in transformations_cfg and\
                isinstance(transformations_cfg.rgb_augmentations, ListConfig):
                self._rgb_augmentations = GSODataModule._set_transformations(
                    transformations_cfg.rgb_augmentations,
                    p=transformations_cfg.augmentations_p,
                )
            
            # Depth augmentations
            if "depth_augmentations" in transformations_cfg and\
                isinstance(transformations_cfg.depth_augmentations, ListConfig):
                self._depth_augmentations = GSODataModule._set_transformations(
                    transformations_cfg.depth_augmentations,
                    p=transformations_cfg.augmentations_p,
                )
            
            # Background augmentations
            if "background_augmentations" in transformations_cfg and\
                isinstance(transformations_cfg.background_augmentations, ListConfig):
                self._background_augmentations = GSODataModule._set_transformations(
                    transformations_cfg.background_augmentations,
                    p=transformations_cfg.augmentations_p,
                )
        
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
        # Load and split datasets only if not loaded already
        if not self._data_train and not self._data_val and not self._data_test:
            
            # Create an iterable scene set from [a/multiple] scene set(s)
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
            self._data_val = data_train
            self._data_test = None

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self._data_train,
            collate_fn=ObjectSegmentationDataset.collate_fn,
            # worker_init_fn=worker_init_fn,
            **self.hparams.dataloader_cfg,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self._data_val,
            collate_fn=ObjectSegmentationDataset.collate_fn,
            # worker_init_fn=worker_init_fn,
            **self.hparams.dataloader_cfg,
        )

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
    
    @staticmethod
    def _set_transformations(
        transformations_list_cfg: ListConfig,
        p: float = 1.0,
    ) -> transformations.SceneObservationTransform:
        """Set the transformations to apply to the data. It composes the transformations
        listed in the configuration.

        Args:
            transformations_list_cfg (ListConfig): List of transformations to compose
                and apply to the data.
            p (float, optional): Probability of applying the transformations. Defaults
                to 1.0.

        Returns:
            transformations.SceneObservationTransform: Composed transformations.
        """
        transformations_list = []
        
        # Parse the configuration for the transformations
        for trans in transformations_list_cfg:
            trans_type = getattr(transformations, trans.type)
            transformations_list.append(trans_type(**trans.params))

        # Compose the transformations
        return transformations.ComposeSceneObservationTransform(
            transformations_list,
            p=p,
        )


if __name__ == "__main__":
    _ = GSODataModule() 
    