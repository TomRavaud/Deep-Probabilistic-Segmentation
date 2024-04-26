# Standard libraries
from typing import Tuple

# Third-party libraries
import torch
import torch.nn as nn
from mobile_sam import sam_model_registry, SamPredictor

import numpy as np
import cv2

# Custom modules
from toolbox.datasets.segmentation_dataset import BatchSegmentationData


class MobileSAM(nn.Module):
    """
    Module that uses the MobileSAM model to predict masks.
    """
    def __init__(self):
        
        super().__init__()
        
        # Choose the image encoder
        model_type = "vit_t"
 
        # Load the MobileSAM weights
        sam_checkpoint = "weights/mobile_sam.pt"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the MobileSAM model and set it to evaluation mode
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        
        # Create the SAM predictor
        self._predictor = SamPredictor(mobile_sam)
    
    @staticmethod
    def _get_bboxes_from_contours(contours: Tuple) -> Tuple:
        """
        Get the bounding box from contours points
        """
        # TODO: Data representation to be changed to avoid for loops
        contours = [contour.squeeze() for contour in contours]
        
        xmin = np.min(contours[0][:, 0])
        xmax = np.max(contours[0][:, 0])
        ymin = np.min(contours[0][:, 1])
        ymax = np.max(contours[0][:, 1])
        
        for contour in contours[1:]:
            xmin = min(xmin, np.min(contour[:, 0]))
            xmax = max(xmax, np.max(contour[:, 0]))
            ymin = min(ymin, np.min(contour[:, 1]))
            ymax = max(ymax, np.max(contour[:, 1]))
        
        return np.array([xmin, ymin, xmax, ymax])
    
    @torch.no_grad()
    def forward(
        self,
        x: BatchSegmentationData,
        contour_points: Tuple,
        hierarchy: np.ndarray,
    ) -> torch.Tensor:
        
        # Get the first image from the batch
        img = x.rgbs[0]
        
        # Copy the image for visualization
        img_original = img.permute(1, 2, 0).cpu().numpy()
        img_to_display = img_original.copy()
        
        # Add the batch dimension
        img = img[None, ...]
        
        # Store the original size of the image before resizing it
        original_size = img.shape[2:]
        
        # Transform the image to the form expected by the model
        img = self._predictor.transform.apply_image_torch(img)
        
        # Send the image to the device
        img = img.to(device=self._predictor.device)
        
        # TODO: Make the prediction on the whole batch
        # Set input images
        self._predictor.set_torch_image(
            img,
            original_image_size=original_size,
        )
        
        
        ### Prompt ###
        # 1. Point-based prediction
        # point_coords = np.array([[150, 120]])
        # point_labels = np.array([1])

        # 2. Box-based prediction
        bbox = MobileSAM._get_bboxes_from_contours(contour_points)
        print("Bounding box: ", bbox)
        
        # 3. Mask-based prediction
        # mask = ...
        # logits = _compute_logits_from_mask(gt_mask)


        # Predict the mask(s) from the image and prompt
        # When multimask_output is True, 3 masks are returned
        masks, scores, logits = self._predictor.predict(
            # point_coords=point_coords,
            # point_labels=point_labels,
            box=bbox,
            # mask_input=logits,
            multimask_output=True,
        )
        
        
        ###########################################################################
        # Debugging
        ###########################################################################
        print("Scores:", scores)
        # print("Logits:", logits.shape)
        
        # Draw the point(s) on the image
        # for point in point_coords:
        #     x, y = point
        #     cv2.circle(img_to_display, (x, y), 5, (0, 0, 255), -1)

        # Draw the bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_to_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get the mask with the highest score
        mask = masks[np.argmax(scores)]

        # Get the logits with the highest score
        # logit = logits[np.argmax(scores)]
        
        
        # Mask the original image
        mask = mask.astype(np.uint8)*255
        
        masked_img = cv2.bitwise_and(img_original, img_original, mask=mask)

        cv2.imshow("Image", img_to_display)
        cv2.imshow("Mask", mask)
        cv2.imshow("Masked image", masked_img)
        cv2.waitKey(0)
        ###########################################################################
        
        return


if __name__ == "__main__":
    
    pass
