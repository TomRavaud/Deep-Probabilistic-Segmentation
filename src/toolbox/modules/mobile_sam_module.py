# Third-party libraries
import torch
import torch.nn as nn
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mobile_sam.utils.transforms import ResizeLongestSide

import numpy as np
import cv2


class MobileSAM(nn.Module):
    """
    Module that uses the MobileSAM model to predict masks.
    """
    def __init__(self):
        pass
    
    def forward(self):
        pass


if __name__ == "__main__":
    
    def _compute_logits_from_mask(mask, eps=1e-3):
        """
        https://github.com/computational-cell-analytics/micro-sam/blob/83997ff4a471cd2159fda4e26d1445f3be79eb08/micro_sam/prompt_based_segmentation.py#L71-L107
        """

        def inv_sigmoid(x):
            return np.log(x / (1 - x))

        logits = np.zeros(mask.shape, dtype="float32")
        logits[mask == 1] = 1 - eps
        logits[mask == 0] = eps
        logits = inv_sigmoid(logits)

        # resize to the expected mask shape of SAM (256x256)
        assert logits.ndim == 2
        expected_shape = (256, 256)

        if logits.shape == expected_shape:  # shape matches, do nothing
            pass

        elif logits.shape[0] == logits.shape[1]:  # shape is square
            trafo = ResizeLongestSide(expected_shape[0])
            logits = trafo.apply_image(logits[..., None])

        else:  # shape is not square
            # resize the longest side to expected shape
            trafo = ResizeLongestSide(expected_shape[0])
            logits = trafo.apply_image(logits[..., None])

            # pad the other side
            h, w = logits.shape
            padh = expected_shape[0] - h
            padw = expected_shape[1] - w
            # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
            pad_width = ((0, padh), (0, padw))
            logits = np.pad(logits, pad_width, mode="constant", constant_values=0)

        logits = logits[None]
        assert logits.shape == (1, 256, 256), f"{logits.shape}"

        return logits


    # Choose the image encoder
    model_type = "vit_t"

    # Load the MobileSAM weights
    sam_checkpoint = "weights/mobile_sam.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the MobileSAM model and set it to evaluation mode
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()


    # Read the image
    img_path = "src/toolbox/modules/frame.png"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Get width and height
    h, w = img.shape[:2]
    # Scale factor
    scale = 2
    # Resize the image
    img = cv2.resize(img, (w*scale, h*scale))

    img_to_display = img.copy()


    # Make a prediction
    predictor = SamPredictor(mobile_sam)

    predictor.set_image(img)

    # Prompt

    # 1. Point-based prediction
    point_coords = scale*np.array([[150, 120]])
    point_labels = np.array([1])

    # 2. Box-based prediction
    box = np.array([150, 80, 450, 330])

    # 3. Mask-based prediction
    gt_mask = cv2.imread("src/toolbox/modules/binary_cat.png", cv2.IMREAD_GRAYSCALE)
    # gt_mask = cv2.resize(gt_mask, (256, 256))
    # gt_mask = gt_mask.astype(np.float32)/255
    gt_mask = np.uint8(gt_mask > 0)


    logits = _compute_logits_from_mask(gt_mask)

    # cv2.imshow("GT Mask", gt_mask)


    # When multimask_output is True, 3 masks are returned
    masks, scores, logits = predictor.predict(
        # point_coords=point_coords,
        # point_labels=point_labels,
        # box=box,
        mask_input=logits,
        multimask_output=True,
    )

    print("Scores:", scores)
    # print("Logits:", logits.shape)

    # Draw the point(s) on the image
    for point in point_coords:
        x, y = point
        cv2.circle(img_to_display, (x, y), 5, (0, 0, 255), -1)

    # Draw the bounding box
    x1, y1, x2, y2 = box
    cv2.rectangle(img_to_display, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # Get the mask with the highest score
    mask = masks[np.argmax(scores)]

    # Get the logits with the highest score
    logit = logits[np.argmax(scores)]


    # #################
    # # Use the predicted mask as input
    # masks, scores, logits = predictor.predict(
    #     # point_coords=point_coords,
    #     # point_labels=point_labels,
    #     # box=box,
    #     mask_input=logit[None, ...],
    #     multimask_output=True,
    # )
    # #################

    # Get the mask with the highest score
    mask = masks[np.argmax(scores)]


    # Mask the original image
    mask = mask.astype(np.uint8)*255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Image", img_to_display)
    cv2.imshow("Mask", mask)
    cv2.imshow("Masked image", masked_img)
    cv2.waitKey(0)
