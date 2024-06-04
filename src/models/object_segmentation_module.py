# Standard libraries
from typing import Any, Dict, Tuple

# Third-party libraries
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

# Custom modules
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


class ObjectSegmentationLitModule(LightningModule):
    
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param model: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
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
        segmentation_masks = self.forward(batch)
        
        # Compute the loss between the model's predictions and the GT masks
        loss = self._criterion(
            segmentation_masks,
            batch.masks,
        )
        
        # NOTE: debugging purposes
        if self.trainer.global_step % 50 == 0:
            idx = 0
            gt = batch.masks[idx].cpu().detach().numpy()
            pred = torch.sigmoid(segmentation_masks[idx]).cpu().detach().numpy()

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
    
    def on_before_optimizer_step(self, optimizer) -> None:
        
        # def plot_grad_flow(named_parameters):
        #     '''Plots the gradients flowing through different layers in the net during training.
        #     Can be used for checking for possible gradient vanishing / exploding problems.

        #     Usage: Plug this function in Trainer class after loss.backwards() as 
        #     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        #     ave_grads = []
        #     max_grads= []
        #     layers = []

        #     for n, p in named_parameters:
        #         if(p.requires_grad) and ("bias" not in n):
        #             layers.append(n)
        #             ave_grads.append(p.grad.abs().mean().cpu())
        #             max_grads.append(p.grad.abs().max().cpu())

        #     # plt.clf()
        #     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        #     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        #     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        #     plt.xlim(left=0, right=len(ave_grads))
        #     # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        #     plt.xlabel("Layers")
        #     plt.ylabel("average gradient")
        #     plt.title("Gradient flow")
        #     plt.grid(True)
        #     plt.legend([Line2D([0], [0], color="c", lw=4),
        #                 Line2D([0], [0], color="b", lw=4),
        #                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        #     plt.savefig("grad_flow.png")
        
        # if self.trainer.global_step % 10 == 0:  # 1 batch out of 10
            
        #     plot_grad_flow(self._model.named_parameters())
        pass
            

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
        pass

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
