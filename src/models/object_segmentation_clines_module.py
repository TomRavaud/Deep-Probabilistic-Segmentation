# Standard libraries
from typing import Any, Dict, Tuple
from functools import partial

# Third-party libraries
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from omegaconf import DictConfig

# Custom modules
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Change font
plt.rcParams.update({"font.family": "serif"})


class ObjectSegmentationCLinesLitModule(LightningModule):
    
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()

        # Allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model", "criterion"])
        
        # Model
        self._model = model

        # Loss function
        self._criterion = criterion
        
        # For averaging loss across batches
        self._train_loss = MeanMetric()
        self._val_loss = MeanMetric()
        self._test_loss = MeanMetric()

        # For tracking best validation loss so far
        self._val_loss_best = MinMetric()

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
        clines_segmentation_masks = self.forward(batch)
        
        # Get the GT masks
        clines_masks = batch.clines_masks
        
        losses = []
        
        # Focus only on the non-padded part of the lines
        for clines_segmentation_mask, clines_mask in zip(
            clines_segmentation_masks,
            clines_masks,
        ):
            # Mask pixels which are inside the borders or which belong to the lines we are
            # not interested in
            mask = clines_mask == 127
            clines_segmentation_mask = clines_segmentation_mask[~mask]
            clines_mask = clines_mask[~mask]
            
            # Check if there are no clines in the image
            if not torch.any(clines_mask):
                continue
            
            # Convert clines_mask to boolean
            clines_mask = clines_mask == 255
            # Convert to float
            clines_mask = clines_mask.float()
            
            losses.append(self._criterion(clines_segmentation_mask, clines_mask))

        loss = torch.mean(torch.stack(losses))
        
        
        # NOTE: debugging purposes
        if self.trainer.global_step % 100 == 0:
            idx = 0
            gt = batch.clines_masks[idx].cpu().detach().numpy()
            pred = torch.sigmoid(clines_segmentation_masks[idx]).cpu().detach().numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(gt)
            plt.title("GT")
            plt.subplot(1, 2, 2)
            plt.imshow(pred)
            plt.title("Pred")
            plt.savefig("result_segmentation.png")
            plt.close()


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
        clines_segmentation_masks = self.forward(batch)
        
        # Get the GT masks
        clines_masks = batch.clines_masks
        
        losses = []
        
        # Focus only on the non-padded part of the lines
        for clines_segmentation_mask, clines_mask in zip(
            clines_segmentation_masks,
            clines_masks,
        ):
            # Mask pixels which are inside the borders or which belong to the lines we
            # are not interested in
            mask = clines_mask == 127
            clines_segmentation_mask = clines_segmentation_mask[~mask]
            clines_mask = clines_mask[~mask]
            
            # Check if there are no clines in the image
            if not torch.any(clines_mask):
                continue
            
            # Convert clines_mask to boolean
            clines_mask = clines_mask == 255
            # Convert to float
            clines_mask = clines_mask.float()
            
            losses.append(self._criterion(clines_segmentation_mask, clines_mask))

        loss = torch.mean(torch.stack(losses))
        
        
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
        clines_segmentation_masks = self.forward(batch)
        
        # Get the GT masks
        clines_masks = batch.clines_masks
        
        losses = []
        
        # Focus only on the non-padded part of the lines
        for clines_segmentation_mask, clines_mask in zip(
            clines_segmentation_masks,
            clines_masks,
        ):
            # Mask pixels which are inside the borders or which belong to the lines we are
            # not interested in
            mask = clines_mask == 127
            clines_segmentation_mask = clines_segmentation_mask[~mask]
            clines_mask = clines_mask[~mask]
            
            # Check if there are no clines in the image
            if not torch.any(clines_mask):
                continue
            
            # Convert clines_mask to boolean
            clines_mask = clines_mask == 255
            # Convert to float
            clines_mask = clines_mask.float()
            
            losses.append(self._criterion(clines_segmentation_mask, clines_mask))

        loss = torch.mean(torch.stack(losses))
        

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
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Define and configure optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: A dictionary containing the configures optimizer(s)
            and learning rate scheduler(s).
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        
        if "scheduler" in self.hparams:
            
            # Manage multiple schedulers
            if isinstance(self.hparams.scheduler, DictConfig)\
                and "main_scheduler" in self.hparams.scheduler\
                and "sub_schedulers" in self.hparams.scheduler:
                    sub_schedulers = [
                        sub_scheduler(optimizer=optimizer)
                        for sub_scheduler in self.hparams.scheduler.sub_schedulers
                    ]
                    scheduler = self.hparams.scheduler.main_scheduler(
                        optimizer=optimizer,
                        schedulers=sub_schedulers)
            elif isinstance(self.hparams.scheduler, partial):
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            else:
                raise ValueError("Invalid scheduler configuration.")
            
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
    pass
