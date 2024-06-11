# Standard libraries
from typing import Tuple, Optional

# Third-party libraries
import torch
import torch.nn as nn
from mobile_sam import sam_model_registry
from mobile_sam.utils.transforms import ResizeLongestSide

import numpy as np
import cv2


class MobileSAM(nn.Module):
    """
    Module that uses the MobileSAM model to predict masks.
    """
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        compile: bool = False,
    ) -> None:
        
        super().__init__()
        
        # Choose the image encoder
        model_type = "vit_t"

        # Load the MobileSAM model and set it to evaluation mode
        self._mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self._mobile_sam.eval()
        
        if compile:
            self._mobile_sam = torch.compile(self._mobile_sam)
        
    @staticmethod
    def _get_bboxes_from_contours(contours: list) -> torch.Tensor:
        """Compute the bounding box from a list of contours

        Args:
            contours (list): List of contours. Each contour is a np.ndarray of
            shape Nx2.

        Returns:
            torch.Tensor: Bounding box in the format [xmin, ymin, xmax, ymax]
        """
        try:
            xmin, ymin = np.min(contours[0], axis=0)
            xmax, ymax = np.max(contours[0], axis=0)
        except IndexError:
            return torch.Tensor([0, 0, 0, 0])
        
        for contour in contours[1:]:
            
            # Get the min and max values of the next contour
            xmin_tmp, ymin_tmp = np.min(contour, axis=0)
            xmax_tmp, ymax_tmp = np.max(contour, axis=0)
            
            # Update the bounding box   
            xmin = min(xmin, xmin_tmp)
            xmax = max(xmax, xmax_tmp)
            ymin = min(ymin, ymin_tmp)
            ymax = max(ymax, ymax_tmp)
        
        return torch.Tensor([xmin, ymin, xmax, ymax])
    
    @torch.no_grad()
    def forward(
        self,
        imgs: torch.Tensor,
        contour_points_list: list[Tuple],
    ) -> torch.Tensor:
        
        # Get the first image for visualization
        img = imgs[0]
         
        # Store the original size of the images before resizing them
        original_size = imgs.shape[2:]
        
        # Set the resizing transformation
        resize_transform = ResizeLongestSide(self._mobile_sam.image_encoder.img_size)

        # The MobileSAM model expects float images
        imgs = imgs.float()
        
        # Resize the images
        imgs = resize_transform.apply_image_torch(imgs)
        
        # Compute the bounding boxes
        bboxes = torch.stack([
            MobileSAM._get_bboxes_from_contours(contour)
            for contour in contour_points_list
        ]).to(device=imgs.device)
        
        # Resize the bounding boxes
        bboxes = resize_transform.apply_boxes_torch(
            bboxes,
            original_size,
        )
        
        # Prepare the (batched) input for the MobileSAM model
        batched_input = [
            {
                "image": imgs[i],
                "boxes": bboxes[i].unsqueeze(0),
                "original_size": original_size,
            }
            for i in range(imgs.shape[0])
        ]
        
        # Get the predictions from the MobileSAM model
        batched_output = self._mobile_sam(batched_input, multimask_output=False)
        
        
        ###########################################################################
        # Debugging
        ########################################################################### 
        
        # # Copy the image for visualization
        # img_original = img.permute(1, 2, 0).cpu().numpy()
        # img_to_display = img_original.copy()
        
        # # RGB to BGR
        # img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        # img_to_display = cv2.cvtColor(img_to_display, cv2.COLOR_RGB2BGR)
        
        # scores = batched_output[0]["iou_predictions"][0].cpu().numpy()
        # # logits = batched_output[0]["low_res_logits"][0].cpu().numpy()
        # masks = batched_output[0]["masks"][0].cpu().numpy()

        
        # print("Scores:", scores)
        # # print("Logits:", logits.shape)
        
        # bbox = MobileSAM._get_bboxes_from_contours(contour_points_list[0])
        
        # # Convert to int
        # bbox = bbox.int().cpu().numpy()
        
        # # Draw the bounding box
        # x1, y1, x2, y2 = bbox
        # cv2.rectangle(img_to_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # # Get the mask with the highest score
        # mask = masks[np.argmax(scores)]
        
        # # Get the logits with the highest score
        # # logit = logits[np.argmax(scores)]
        
        # # Mask the original image
        # mask = mask.astype(np.uint8)*255
        
        # masked_img = cv2.bitwise_and(img_original, img_original, mask=mask)

        # cv2.imshow("Image", img_to_display)
        # # cv2.imshow("Mask", mask)
        # cv2.imshow("Masked image", masked_img)
        # cv2.waitKey(0)
        ###########################################################################
        
        return batched_output


if __name__ == "__main__":
    pass
