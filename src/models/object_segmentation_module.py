# Standard libraries
from typing import Any, Dict, Tuple

# Third-party libraries
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

# Custom modules
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


class ObjectSegmentationLitModule(LightningModule):
    """A PyTorch Lightning module for object segmentation.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param model: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # Allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model"])

        self._model = model

        # TODO: try focal loss and dice loss
        # Loss function
        self._criterion = torch.nn.CrossEntropyLoss()

        # For averaging loss across batches
        self._train_loss = MeanMetric()
        self._val_loss = MeanMetric()
        self._test_loss = MeanMetric()

        # For tracking best validation loss so far
        self._val_loss_best = MaxMetric()

    def forward(self, x: BatchSegmentationData) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (BatchSegmentationData): A batch of data.

        Returns:
            torch.Tensor: The model's predictions.
        """
        return self._model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training
        # starts, so it's worth to make sure validation metrics don't store results
        # from these checks
        self._val_loss.reset()
        self._val_loss_best.reset()

    def training_step(
        self,
        batch: BatchSegmentationData,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        
        # Compute the output of the model
        segmentation_masks = self.forward(batch)
        
        # Compute the loss between the model's predictions and the GT masks
        loss = self._criterion(
            segmentation_masks,
            batch.masks,
        )
        
        # Update and log metric (average loss across batches)
        self._train_loss(loss)
        self.log(
            "train/loss",
            self._train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self,
        batch: BatchSegmentationData,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation
        set.
        """
        # Compute the output of the model
        segmentation_masks = self.forward(batch)
        
        # Compute the loss between the model's predictions and the GT masks
        loss = self._criterion(
            segmentation_masks,
            batch.masks,
        )

        # Update and log metric
        self._val_loss(loss)
        self.log(
            "val/loss",
            self._val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_loss = self._val_loss.compute()  # get current val loss
        self._val_loss_best(val_loss)  # update best so far val loss
        self.log(
            "val/loss_best",
            self._val_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Perform a single test step on a batch of data from the test set.
        """
        # Compute the output of the model
        segmentation_masks = self.forward(batch)
        
        # Compute the loss between the model's predictions and the GT masks
        loss = self._criterion(
            segmentation_masks,
            batch.masks,
        )

        # Update and log metric
        self._test_loss(loss)
        self.log(
            "test/loss",
            self._test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate),
        validate, test, or predict.
        """
        if self.hparams.compile and stage == "fit":
            self._model = torch.compile(self._model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Define and configure optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: A dictionary containing the configures optimizer(s)
            and learning rate scheduler(s).
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ObjectSegmentationLitModule(None, None, None, None)
