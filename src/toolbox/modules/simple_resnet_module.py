# Standard libraries
from typing import Optional

# Third party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchinfo


class SimpleResNet(nn.Module):
    """
    ResNet model.
    """
    def __init__(
        self,
        version: int = 18,
        output_dim: tuple = (1000,),
        nb_input_channels: Optional[int] = None,
        output_logits: bool = True,
    ) -> None:
        """Initialize a pretrained `ResNet` module.

        Args:
            version (int, optional): The version of the ResNet model to use.
            output_dim (int, optional): Dimension of the output. Defaults to (1000,).
            nb_input_channels (Optional[int], optional): Number of input channels.
                If None, the number of input channels is not fixed and can be set at
                runtime. If not None, the number of input channels is fixed to the
                specified value. Defaults to None.
            output_logits (bool, optional): Whether to output logits or probabilities.
                Defaults to True.
        """
        super(SimpleResNet, self).__init__()
        
        # Load the ResNet model with pretrained weights
        if version == 18:
            self._resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif version == 34:
            self._resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif version == 50:
            self._resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported ResNet version: {version}. "
                             "It may be an invalid version number or the "
                             "version is not supported yet. "
                             "Supported versions are 18, 34, and 50.")
        
        if nb_input_channels is None:
            # Replace the first convolutional layer by a lazy one to allow for dynamic
            # input channels 
            self._resnet.conv1 = nn.LazyConv2d(
                out_channels=self._resnet.conv1.out_channels,
                kernel_size=self._resnet.conv1.kernel_size,
                stride=self._resnet.conv1.stride,
                padding=self._resnet.conv1.padding,
                bias=self._resnet.conv1.bias,
            )
        elif nb_input_channels != 3:
            # Replace the first convolutional layer by a convolutional layer with the
            # desired number of input channels
            self._resnet.conv1 = nn.Conv2d(
                in_channels=nb_input_channels,
                out_channels=self._resnet.conv1.out_channels,
                kernel_size=self._resnet.conv1.kernel_size,
                stride=self._resnet.conv1.stride,
                padding=self._resnet.conv1.padding,
                bias=self._resnet.conv1.bias,
            )
        
        output_size = sum(output_dim)
        
        if output_size != 1000:
            # Replace the last fully-connected layer to have output_size
            # classes as output
            self._resnet.fc = nn.Linear(
                in_features=self._resnet.fc.in_features,
                out_features=output_size,
                bias=self._resnet.fc.bias is not None,
            )

        # Apply the sigmoid activation function to the output in inference mode only
        # (not during training because it is applied in the loss function to ensure
        # numerical stability)
        if output_logits:
            self._output_activation = nn.Identity()
        else:
            self._output_activation = nn.Sigmoid()
        
        self._output_dim = output_dim
    
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
        # Forward pass through ResNet
        x = self._resnet.conv1(x)
        x = self._resnet.bn1(x)
        x = self._resnet.relu(x)
        x = self._resnet.maxpool(x)
        
        x = self._resnet.layer1(x)
        x = self._resnet.layer2(x)
        x = self._resnet.layer3(x)
        x = self._resnet.layer4(x)
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
        x = self._resnet.fc(x)
        
        # Sigmoid or Identity activation function
        x = self._output_activation(x)
        
        # Reshape the output to match the expected output shape
        if len(self._output_dim) > 1:
            x = x.view(-1, *self._output_dim)
        
        return x


if __name__ == "__main__":
    
    torchinfo.summary(
        SimpleResNet(
            version=34,
            output_dim=(15,),
            nb_input_channels=None,
        ),
        input_size=(32, 5, 224, 224),
    )
