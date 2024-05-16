"""
This module contains the implementation of the Dice loss, Focal loss, and the Focal-Dice
combination loss.
"""
# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Implementation of the Dice loss. The Dice loss is a metric that measures the overlap
    between two samples, as the IoU metric does, with the difference that it is
    differentiable. In a segmentation task, it encourages a better high-level spatial
    overlap between the predicted and true segmentation masks.
    """
    def __init__(self, smooth: float = 1.0, with_logits: bool = True) -> None:
        """Constructor of the class.

        Args:
            smooth (float, optional): A smoothing factor used to prevent division by
                zero and reduce the risk of overfitting. The higher the value, the
                smoother the loss. Defaults to 1.0.
            with_logits (bool, optional): A flag that determines if the input is passed
                through a sigmoid function before computing the loss. Defaults to False.
        """
        super().__init__()
        
        self._smooth = smooth
        self._with_logits = with_logits
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss between the predicted and true segmentation masks.
        
        Args:
            y_pred: The predicted segmentation masks.
            y_true: The true segmentation masks.
        
        Returns:
            The Dice loss between the predicted and true segmentation masks.
        """
        # Apply the sigmoid function to scale the predictions between 0 and 1
        if self._with_logits:
            y_pred = torch.sigmoid(y_pred)
        
        # Compute the intersection between the predicted and true masks
        intersection = (y_pred * y_true).sum()
        
        # Compute the union between the predicted and true masks
        union = (y_pred).sum() + (y_true).sum()
        
        # Compute the Dice loss
        dice = 1 - (2 * intersection + self._smooth) / (union + self._smooth)
        
        return dice


class FocalLoss(nn.Module):
    """
    Implementation of the Focal loss. The Focal loss is a modified version of the
    Cross-Entropy loss that focuses on hard examples. In a segmentation task, it
    encourages low-level pixel accuracy between the predicted and true segmentation
    masks.
    
    Inspired by the official implementation:
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py#L9-L51
    """
    def __init__(
        self,
        gamma: float = 2,
        alpha: float = 0.25,
        with_logits: bool = True,
    ) -> None:
        """Constructor of the class.

        Args:
            gamma (float, optional): A focusing parameter that determines how much the
                loss focuses on hard examples. The higher the value, the more the loss
                focuses on hard examples. Defaults to 2.
            alpha (float, optional): A balancing parameter that determines the weight of
                the positive class. The higher the value, the more the loss focuses on
                the positive class. Defaults to 0.25.
            with_logits (bool, optional): A flag that determines if the input is passed
                through a sigmoid function before computing the loss. Defaults to False.
        """
        super().__init__()
        
        self._gamma = gamma
        self._alpha = alpha
        self._with_logits = with_logits
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the Focal loss between the predicted and true segmentation masks.
        
        Args:
            y_pred: The predicted segmentation masks.
            y_true: The true segmentation masks.
        
        Returns:
            The Focal loss between the predicted and true segmentation masks.
        """
        # Compute the binary cross-entropy loss
        if self._with_logits:
            # Combine the sigmoid function and binary cross-entropy loss to take
            # advantage of the log-sum-exp trick for numerical stability
            bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
            
            # Apply the sigmoid function to scale the predictions between 0 and 1
            y_pred = torch.sigmoid(y_pred)
        
        else:
            bce = F.binary_cross_entropy(y_pred, y_true, reduction="none")
        
        # Compute the focal loss
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = (1 - pt) ** self._gamma * bce
        
        # Imbalance class weighting
        if self._alpha >= 0:
            alpha_t = self._alpha * y_true + (1 - self._alpha) * (1 - y_true)
            focal *= alpha_t
        
        return focal.mean()


class FocalDiceCombinationLoss(nn.Module):
    """
    Implementation of the Focal-Dice combination loss.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        smooth: float = 1.0,
        with_logits: bool = True,
        k_focal: float = 20.0,
        k_dice: float = 1.0,
    ) -> None:
        """Constructor of the class.
        
        Args:
            gamma (float, optional): A focusing parameter that determines how much the
                Focal loss focuses on hard examples. The higher the value, the more the
                loss focuses on hard examples. Defaults to 2.0.
            alpha (float, optional): A balancing parameter that determines the weight of
                the positive class in the Focal loss. The higher the value, the more the
                loss focuses on the positive class. Defaults to 0.25.
            smooth (float, optional): A smoothing factor used to prevent division by
                zero and reduce the risk of overfitting in the Dice loss. The higher the
                value, the smoother the loss. Defaults to 1.0.
            with_logits (bool, optional): A flag that determines if the input is passed
                through a sigmoid function before computing the loss. Defaults to True.
            k_focal (float, optional): A balancing parameter that determines the weight
                of the Focal loss. The higher the value, the more the loss focuses on
                the Focal loss. Defaults to 20.0.
            k_dice (float, optional): A balancing parameter that determines the weight
                of the Dice loss. The higher the value, the more the loss focuses on
                the Dice loss. Defaults to 1.0.
        """
        super().__init__()
        
        # Instantiate the Focal and Dice losses
        self._focal_loss = FocalLoss(gamma, alpha, with_logits)
        self._dice_loss = DiceLoss(smooth, with_logits)
        
        self._k_focal = k_focal
        self._k_dice = k_dice
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the Focal and Dice loss between the predicted and true segmentation
        masks.
        
        Args:
            y_pred: The predicted segmentation masks.
            y_true: The true segmentation masks.
        
        Returns:
            The Focal and Dice loss between the predicted and true segmentation masks.
        """
        focal_loss = self._focal_loss(y_pred, y_true)
        dice_loss = self._dice_loss(y_pred, y_true)
        
        return self._k_focal * focal_loss + self._k_dice * dice_loss


if __name__ == "__main__":
    
    # Test
    y_pred = torch.tensor(
        [
            [
                [0.1, 0.8],
                [0.2, 0.9],
            ],
        ]
    )
    y_true = torch.tensor(
        [
            [
                [0., 1.],
                [0., 1.],
            ],
        ]
    )
    
    focal_dice_loss = FocalDiceCombinationLoss(with_logits=False)
    loss = focal_dice_loss(y_pred, y_true)
    print("Focal-Dice loss:", loss.item())
