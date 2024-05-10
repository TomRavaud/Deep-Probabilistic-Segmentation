# Standard libraries
from typing import Optional

# Third party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchinfo


class ResNet18(nn.Module):
    """
    ResNet18 model.
    """
    def __init__(
        self,
        output_dim: int = 1000,
        nb_input_channels: Optional[int] = None,
    ) -> None:
        """Initialize a pretrained `ResNet18` module.

        Args:
            output_dim (int, optional): Dimension of the output. Defaults to 10.
            nb_input_channels (Optional[int], optional): Number of input channels.
                If None, the number of input channels is not fixed and can be set at
                runtime. If not None, the number of input channels is fixed to the
                specified value. Defaults to None.
        """
        super(ResNet18, self).__init__()
        
        # Load the ResNet18 model with pretrained weights
        self._resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        if nb_input_channels is None:
            # Replace the first convolutional layer by a lazy one to allow for dynamic
            # input channels 
            self._resnet18.conv1 = nn.LazyConv2d(
                out_channels=self._resnet18.conv1.out_channels,
                kernel_size=self._resnet18.conv1.kernel_size,
                stride=self._resnet18.conv1.stride,
                padding=self._resnet18.conv1.padding,
                bias=self._resnet18.conv1.bias,
            )
        elif nb_input_channels != 3:
            # Replace the first convolutional layer by a convolutional layer with the
            # desired number of input channels
            self._resnet18.conv1 = nn.Conv2d(
                in_channels=nb_input_channels,
                out_channels=self._resnet18.conv1.out_channels,
                kernel_size=self._resnet18.conv1.kernel_size,
                stride=self._resnet18.conv1.stride,
                padding=self._resnet18.conv1.padding,
                bias=self._resnet18.conv1.bias,
            )
        
        if output_dim != 1000:
            # Replace the last fully-connected layer to have output_dim
            # classes as output
            self._resnet18.fc = nn.Linear(
                in_features=self._resnet18.fc.in_features,
                out_features=output_dim,
                bias=self._resnet18.fc.bias is not None,
            )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        # Forward pass through ResNet18
        x = self._resnet18.conv1(x)
        x = self._resnet18.bn1(x)
        x = self._resnet18.relu(x)
        x = self._resnet18.maxpool(x)
        
        x = self._resnet18.layer1(x)
        x = self._resnet18.layer2(x)
        x = self._resnet18.layer3(x)
        x = self._resnet18.layer4(x)
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
        x = self._resnet18.fc(x)
        
        # NOTE: The sigmoid activation function is not applied here because it is
        # applied in the loss function (BCEWithLogitsLoss) to ensure numerical
        # stability
        # Apply sigmoid activation function to the output to ensure that the values
        # is between 0 and 1
        # x = torch.sigmoid(x)
        
        return x


if __name__ == "__main__":
    
    torchinfo.summary(
        ResNet18(
            output_dim=15,
            nb_input_channels=None,
        ),
        input_size=(32, 5, 224, 224),
    )
