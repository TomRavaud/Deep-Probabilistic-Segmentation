# Standard libraries
from typing import Optional

# Third-party libraries
import torch
from torch import nn
from omegaconf import DictConfig, ListConfig
import cv2

# Custom modules
from toolbox.evaluation.sequence_segmentation_dataset import (
    BatchSequenceSegmentationData,
)
from toolbox.datasets.make_sets import make_object_set
from toolbox.modules.contour_rendering_module import ContourRendering
from toolbox.modules.mobile_sam_module import MobileSAM
from toolbox.evaluation.mask_rendering_module import MaskRendering


class SequenceSegmentationPredictionModel(nn.Module):
    
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
        image_size: ListConfig,
        sam_checkpoint: Optional[str] = None,
        object_set_cfg: Optional[DictConfig] = None,
        compile: bool = False,
    ) -> None:
        
        super().__init__()
        
        # Create the set of objects
        object_set = make_object_set(**object_set_cfg)
        
        # Instantiate the mask rendering module
        self._mask_rendering_module = MaskRendering(
            object_set=object_set,
            image_size=tuple(image_size),
            debug=True,
        )
        
        # # Instantiate the contour rendering module
        # # (for rendering 3D objects, and extracting points along objects contour)
        # self._contour_rendering_module = ContourRendering(
        #     object_set=object_set,
        #     image_size=tuple(image_size),
        #     debug=True,
        # )
        
        # # Instantiate the MobileSAM module
        # # (for explicit object segmentation alignment)
        # self._mobile_sam = MobileSAM(
        #     sam_checkpoint=sam_checkpoint,
        #     compile=compile,
        # )
        # # Freeze the MobileSAM parameters
        # for param in self._mobile_sam.parameters():
        #     param.requires_grad = False
        
        # self._probabilistic_segmentation_model = probabilistic_segmentation_model
        
        
    def forward(self, x: BatchSequenceSegmentationData) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x (BatchSegmentationData): A batch of segmentation data.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        if x.batch_size != 1:
            raise NotImplementedError(
                "Batch sizes different from 1 are not supported yet."
            )
        
        # Compute the ground truth masks by rendering the objects
        ground_truth_masks = self._mask_rendering_module(x)
        
        
        # # Plot the first image of the sequence
        # img = x.rgbs[0, 0]
        # img = img.permute(1, 2, 0).cpu().numpy()
        
        # mask = ground_truth_masks[0].cpu().numpy()

        # # Image to BGR
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # cv2.imshow("Image", img)
        # cv2.imshow("Mask", mask)
        
        # cv2.waitKey(0)
        
        
        
        # # Render objects of the batch, extract outer contours points
        # contour_points_list = self._contour_rendering_module(x)

        # # Predict masks, scores and logits using the MobileSAM model
        # mobile_sam_outputs = self._mobile_sam(x, contour_points_list)

        # # Stack the masks from the MobileSAM outputs
        # binary_masks = torch.stack([
        #     output["masks"][:, torch.argmax(output["iou_predictions"])]
        #     for output in mobile_sam_outputs
        # ])
        
        # # Get RGB images
        # rgb_images = x.rgbs
        
        # # Compute the probabilistic segmentation masks
        # probabilistic_masks = self._probabilistic_segmentation_model(
        #     rgb_images,
        #     binary_masks,
        # )
        
        # return probabilistic_masks
        return


# class SequenceSegmentationPredictionModule(nn.Module):
    
#     def __init__(
#         self,
#         model: SequenceSegmentationPredictionModel,
#         criterion: nn.Module,
#     ) -> None:
        
#         super().__init__()
        
#         self._model = model
#         self._criterion = criterion
    
#     @torch.no_grad()
#     def forward(self, x: BatchSequenceSegmentationData) -> torch.Tensor:
        
#         # Get the predictions
#         predictions = self._model(x)
        
#         # Compute the loss
#         loss = self._criterion(predictions, x.masks)
        
#         return predictions, loss
        
        


if __name__ == "__main__":
    pass
