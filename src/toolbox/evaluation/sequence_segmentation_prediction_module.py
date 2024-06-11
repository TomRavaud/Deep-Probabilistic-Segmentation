# Standard libraries
from typing import Optional

# Third-party libraries
import torch
from torch import nn
from omegaconf import DictConfig, ListConfig
import numpy as np
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Custom modules
from toolbox.evaluation.sequence_segmentation_dataset import (
    BatchSequenceSegmentationData,
)
from toolbox.datasets.segmentation_dataset import BatchSegmentationData
from toolbox.datasets.scene_set import ObjectData
from toolbox.datasets.make_sets import make_object_set
from toolbox.modules.mobile_sam_module import MobileSAM
from toolbox.modules.mask_rendering_module import MaskRendering


class SequenceSegmentationPredictionModel(nn.Module):
    
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
        image_size: ListConfig,
        sam_checkpoint: Optional[str] = None,
        segmentation_model_checkpoint: Optional[str] = None,
        object_set_cfg: Optional[DictConfig] = None,
        error_metric: nn.Module = JaccardIndex(task="binary"),
        compile: bool = False,
    ) -> None:
        """Constructor.

        Args:
            probabilistic_segmentation_model (nn.Module): The implicit probabilistic
                segmentation model.
            image_size (ListConfig): The size of the input images.
            sam_checkpoint (Optional[str], optional): Pre-trained MobileSAM parameters.
                Defaults to None.
            segmentation_model_checkpoint (Optional[str], optional): Pre-trained
                probabilistic segmentation model parameters. Defaults to None.
            object_set_cfg (Optional[DictConfig], optional): Configuration parameters
                for the object set. Defaults to None.
            error_metric (nn.Module, optional): The error metric to use for the
                evaluation of the probabilistic segmentation model. Defaults to
                JaccardIndex(task="binary").
            compile (bool, optional): Whether to compile the MobileSAM module. Defaults
                to False.
        """
        super().__init__()
        
        # Create the set of objects
        object_set = make_object_set(**object_set_cfg)
        
        # Instantiate the mask rendering module
        self._mask_rendering_module = MaskRendering(
            object_set=object_set,
            image_size=tuple(image_size),
            debug=True,
        )
        
        # Instantiate the MobileSAM module
        # (for explicit object segmentation alignment)
        self._mobile_sam = MobileSAM(
            sam_checkpoint=sam_checkpoint,
            compile=compile,
        )
        # Freeze the MobileSAM parameters
        for param in self._mobile_sam.parameters():
            param.requires_grad = False
        
        # Instantiate the probabilistic segmentation model
        self._probabilistic_segmentation_model = probabilistic_segmentation_model
        
        # Load the weights and biases of the probabilistic segmentation model
        if segmentation_model_checkpoint is not None:
            full_state_dict = torch.load(segmentation_model_checkpoint)["state_dict"]
            
            # Select the keys of the probabilistic segmentation model
            segmentation_model_state_dict = {}
            
            for key, value in full_state_dict.items():
                segmentation_model_state_dict[
                    key.replace("_model._probabilistic_segmentation_model.", "")
                ] = value
        
            # Load the state dict
            self._probabilistic_segmentation_model.load_state_dict(
                segmentation_model_state_dict
            )
        else:
            print("No checkpoint provided for the probabilistic segmentation model.")
            
        # Set the error metric
        self._error_metric = error_metric
    
    @torch.no_grad()
    def forward(self, x: BatchSequenceSegmentationData) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x (BatchSegmentationData): A batch of segmentation data.

        Returns:
            torch.Tensor: A tensor of predictions.
        
        Raises:
            NotImplementedError: If the batch size is different from 1.
            ValueError: If no object pixels are found in the first frame.
        """
        if x.batch_size != 1:
            raise NotImplementedError(
                "Batch sizes different from 1 are not supported yet."
            )
        
        # Set the input data for the mask rendering module
        batch_segmentation_data = BatchSegmentationData(
            rgbs=x.rgbs[0],
            masks=None,
            object_datas=[
                ObjectData(
                    label=x.object_labels[0]
                ) for _ in range(x.sequence_size)],
            bboxes=None,
            TCO=x.TCO[0],
            DTO=None,
            K=x.K.expand(x.sequence_size, -1, -1),
            depths=None,
        )
        
        # Compute the ground truth masks by rendering the objects
        ground_truth_masks = self._mask_rendering_module(batch_segmentation_data)
        
        # Get the sequence of RGB images
        rgb_images = x.rgbs[0]
        
        # Get the first image of the sequence
        first_image = rgb_images[0:1]
        
        # Get the first ground truth mask
        first_ground_truth_mask = ground_truth_masks[0]
        
        # Set the bounding box coordinates for the first frame of the sequence
        indices = torch.nonzero(first_ground_truth_mask)
        
        if len(indices) == 0:
            raise ValueError("No object pixels found in the first frame.")
        else:
            min_coords, _ = torch.min(indices, dim=0)
            max_coords, _ = torch.max(indices, dim=0)
            
            bbox = torch.tensor([
                min_coords[1].item(),
                min_coords[0].item(),
                max_coords[1].item(),
                max_coords[0].item(),
            ])
        
        # Set the MobileSAM expected input
        contour_points_list=[
            # First example of the batch
            [np.array(bbox).reshape(-1, 2),],
            # Second example of the batch...
        ]
        
        # Predict mask for the first image
        mobile_sam_outputs = self._mobile_sam(first_image, contour_points_list)
        
        # Stack the mask(s) from the MobileSAM outputs
        binary_masks = torch.stack([
            output["masks"][:, torch.argmax(output["iou_predictions"])]
            for output in mobile_sam_outputs
        ])
        
        # Compute the probabilistic segmentation mask for the first image
        # (parameters of the implicit segmentation model are set internally)
        first_probabilistic_mask = self._probabilistic_segmentation_model(
            first_image,
            binary_masks,
        )
        
        # Use the segmentation model with parameters set for the first image to
        # predict the masks for the rest of the sequence
        other_probabilistic_masks =\
            self._probabilistic_segmentation_model.forward_pixel_segmentation(
                rgb_images[1:],
            )
        
        # Stack the probabilistic masks
        probabilistic_masks = torch.cat([
            first_probabilistic_mask,
            other_probabilistic_masks,
        ])
        
        # Compute the segmentation error
        error = self._error_metric(
            preds=probabilistic_masks,
            target=ground_truth_masks,
        )
        
        ##### Debugging #####
        # # Plot the first image of the sequence
        # img = x.rgbs[0, 0]
        # img = img.permute(1, 2, 0).cpu().numpy()
        
        # # mask = ground_truth_masks[0].cpu().numpy()
        # mask = binary_masks[0, 0].to(dtype=torch.float32).cpu().numpy()
        
        # probabilistic_mask = first_probabilistic_mask[0].cpu().numpy()

        
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # divider1 = make_axes_locatable(axes[1])
        # cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        # divider2 = make_axes_locatable(axes[2])
        # cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        
        # axes[0].imshow(img)
        # axes[0].set_title("Image")
        # axes[0].axis("off")
        
        # img1 = axes[1].imshow(mask, cmap="magma")
        # axes[1].set_title("GT mask")
        # axes[1].axis("off")
        
        # axes[2].imshow(probabilistic_mask, cmap="magma")
        # axes[2].set_title("Probabilistic mask")
        # axes[2].axis("off")
        
        # # Colorbar
        # fig.colorbar(img1, ax=axes[1], cax=cax1)
        # fig.colorbar(img1, ax=axes[2], cax=cax2)
        
        # plt.tight_layout()
        # plt.savefig("debug.png")
        # plt.close()
        
        
        return error


if __name__ == "__main__":
    pass
