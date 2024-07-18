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
        
        # Get the GT masks
        masks = batch.masks
        
        # Find bounding boxes from the GT masks
        def find_bboxes(masks):
            bboxes = []
            for mask in masks:
                non_zero_indices = torch.nonzero(mask)
                y_min, x_min = torch.min(non_zero_indices, dim=0)[0]
                y_max, x_max = torch.max(non_zero_indices, dim=0)[0]
                bboxes.append([y_min, x_min, y_max, x_max])
            return torch.tensor(bboxes)

        bboxes = find_bboxes(masks)
        
        # Increase the bounding boxes by a factor of 2
        def increase_bboxes(bboxes, height, width, scale=2):
            center_y = (bboxes[:, 0] + bboxes[:, 2]) / 2
            center_x = (bboxes[:, 1] + bboxes[:, 3]) / 2
            half_height = (bboxes[:, 2] - bboxes[:, 0]) / 2 * scale
            half_width = (bboxes[:, 3] - bboxes[:, 1]) / 2 * scale

            y_min = torch.clamp(center_y - half_height, 0, height)
            x_min = torch.clamp(center_x - half_width, 0, width)
            y_max = torch.clamp(center_y + half_height, 0, height)
            x_max = torch.clamp(center_x + half_width, 0, width)

            return torch.stack([y_min, x_min, y_max, x_max], dim=1).long()

        height, width = masks.shape[1], masks.shape[2]
        increased_bboxes = increase_bboxes(bboxes, height, width)
        
        # Extract the bounding boxes from the masks and the predicted masks
        def extract_bbox_regions(masks, bboxes):
            regions = []
            for mask, bbox in zip(masks, bboxes):
                y_min, x_min, y_max, x_max = bbox
                regions.append(mask[y_min:y_max, x_min:x_max])
            return regions

        mask_regions = extract_bbox_regions(masks, increased_bboxes)
        pred_mask_regions = extract_bbox_regions(segmentation_masks, increased_bboxes)
        
        losses = []
        for pred_region, mask_region in zip(pred_mask_regions, mask_regions):
            losses.append(self._criterion(pred_region, mask_region))

        loss = torch.mean(torch.stack(losses))
        
        # Compute the loss between the model's predictions and the GT masks
        # loss = self._criterion(
        #     segmentation_masks,
        #     masks,
        # )
        
        # NOTE: debugging purposes
        if self.trainer.global_step % 300 == 0:
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
        #             # layers.append(n.split(".")[3:])
        #             layers.append(".".join(n.split(".")[3:-1]))
        #             ave_grads.append(p.grad.abs().mean().cpu())
        #             max_grads.append(p.grad.abs().max().cpu())

        #     plt.figure("fig", figsize=(15, 6))
        #     plt.clf()
        #     # plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        #     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=1, lw=1, color="b")
        #     # plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=8)
        #     plt.xlim(left=0, right=len(ave_grads))
        #     # plt.ylim(bottom=0, top=1) # zoom in on the lower gradient regions
        #     plt.xlabel("Layers")
        #     plt.ylabel("Average gradient")
        #     # plt.title("Gradient flow")
        #     plt.grid(True)
        #     # plt.legend(
        #     #     [
        #     #         # Line2D([0], [0], color="c", lw=4),
        #     #         Line2D([0], [0], color="b", lw=4),
        #     #         # Line2D([0], [0], color="k", lw=4)
        #     #     ],
        #     #     [
        #     #         # 'max-gradient',
        #     #         'mean-gradient',
        #     #         # 'zero-gradient'
        #     #     ])
        #     plt.savefig(
        #         f"grad_flow_{self.trainer.global_step}.pgf",
        #         bbox_inches='tight',
        #         pad_inches=0,
        #         transparent=True,
        #     )
        
        # if self.trainer.global_step % 10 == 0:  # 1 batch out of 10
        #      plot_grad_flow(self._model.named_parameters())
             
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
    _ = ObjectSegmentationLitModule(None, None, None, None)
