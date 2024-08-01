"""
Lightning component used to load and prepare the data on which to train, validate, and
test the models.
"""
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
    """  
    def __init__(
        self,
        scene_sets_cfg: Optional[DictConfig] = None,
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
                if transformations_cfg.resize.object_focus:
                    self._resize_transform =\
                        transformations.CropResizeToObjectTransform(
                            resize=transformations_cfg.resize.size,
                        )
                else:
                    self._resize_transform =\
                        transformations.CropResizeToAspectTransform(
                            resize=transformations_cfg.resize.size,
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
        Prepare data. This method is called on 1 GPU/TPU in distributed training.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data for the training, validation, and testing dataloaders.

        Args:
            stage (Optional[str], optional): The stage to set up. Either `"fit"`,
                `"validate"`, `"test"`, or `"predict"`. Defaults to None.
        """
        # Load and split datasets only if not loaded already
        if not self._data_train and not self._data_val and not self._data_test:
            
            # Create iterable scene sets from [a/multiple] scene set(s)
            scene_set_train = make_iterable_scene_set(
                **self.hparams.scene_sets_cfg.train,
            )
            scene_set_val = make_iterable_scene_set(
                **self.hparams.scene_sets_cfg.val,
            )
            scene_set_test = make_iterable_scene_set(
                **self.hparams.scene_sets_cfg.test,
            )
            
            # Datasets
            self._data_train = ObjectSegmentationDataset(
                scene_set_train,
                resize_transform=self._resize_transform,
                background_augmentations=self._background_augmentations,
                rgb_augmentations=self._rgb_augmentations,
                depth_augmentations=self._depth_augmentations,
                **self.hparams.dataset_cfg,
            )
            self._data_val = ObjectSegmentationDataset(
                scene_set_val,
                resize_transform=self._resize_transform,
                background_augmentations=self._background_augmentations,
                rgb_augmentations=self._rgb_augmentations,
                depth_augmentations=self._depth_augmentations,
                **self.hparams.dataset_cfg,
            )
            self._data_test = ObjectSegmentationDataset(
                scene_set_test,
                resize_transform=self._resize_transform,
                background_augmentations=self._background_augmentations,
                rgb_augmentations=self._rgb_augmentations,
                depth_augmentations=self._depth_augmentations,
                **self.hparams.dataset_cfg,
            )
            
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            DataLoader[Any]: The train dataloader.
        """
        return DataLoader(
            dataset=self._data_train,
            collate_fn=ObjectSegmentationDataset.collate_fn,
            **self.hparams.dataloader_cfg,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            DataLoader[Any]: The validation dataloader.
        """
        return DataLoader(
            dataset=self._data_val,
            collate_fn=ObjectSegmentationDataset.collate_fn,
            **self.hparams.dataloader_cfg,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            DataLoader[Any]: The test dataloader.
        """
        return DataLoader(
            dataset=self._data_test,
            collate_fn=ObjectSegmentationDataset.collate_fn,
            **self.hparams.dataloader_cfg,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Called at the end of training, validation, test, or predict. Use for
        cleaning up things and saving files.

        Args:
            stage (Optional[str], optional): The stage being torn down. Either `"fit"`,
                `"validate"`, `"test"`, or `"predict"`. Defaults to None.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Implement to return the datamodule state to save in a checkpoint.

        Returns:
            Dict[Any, Any]: A dictionary containing the datamodule state that you want
                to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Implement to load the datamodule state from a checkpoint.

        Args:
            state_dict (Dict[str, Any]): The datamodule state returned by
                `self.state_dict()`.
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
