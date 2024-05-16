import torch
import torch.nn as nn
import torchinfo


class PixelSegmentationMLP(nn.Module):
    """
    Module that predicts the probability of pixels in an image being part of the
    foreground based on the RGB values of the patches centered at the pixels with a
    multi-layer perceptron (MLP).
    """
    def __init__(
        self,
        patch_size: int = 5,
        nb_channels: int = 3,
        hidden_dims: list[int] = [128, 64, 32],
    ) -> None:
        """Constructor of the class.

        Args:
            patch_size (int, optional): Side length of the square patch. Defaults to 5.
            nb_channels (int, optional): Number of channels in the input tensor.
                Defaults to 3.
            hidden_dims (list[int], optional): Number of hidden units in each layer of
                the MLP. Defaults to [128, 64, 32].
        """
        super(PixelSegmentationMLP, self).__init__()
        
        # Define the architecture of the network
        layers = []
        in_features = patch_size ** 2 * nb_channels
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Instantiate the network
        self._mlp = nn.Sequential(*layers)
        
        # Freeze the network (paramters will be set externally)
        # for param in self._mlp.parameters():
        #     param.requires_grad = False
        
        # Compute the total number of parameters
        self._nb_parameters = sum(p.numel() for p in self._mlp.parameters())
        
        self._patch_size = patch_size
        self._nb_channels = nb_channels
    
    @property
    def nb_parameters(self) -> int:
        """Get the total number of parameters in the module.

        Returns:
            int: Total number of parameters.
        """
        return self._nb_parameters
    
    def set_model_parameters(self, params: torch.Tensor) -> None:
        """Set the parameters of the model with the values in the input tensor.
        
        Args:
            params (torch.Tensor): Tensor containing the parameters of the model.
            
        Raises:
            ValueError: If the number of values in the tensor does not match the number
                of parameters in the model.
        """
        # Ensure the tensor has the correct number of values
        if params.numel() != self._nb_parameters:
            raise ValueError(
                "The number of tensor values must match the number of model parameters"
            )

        # Ensure the tensor is a 1D tensor
        params = params.flatten()
        
        # Assign values from the tensor to model parameters
        idx = 0
        for param in self._mlp.parameters():
            param_numel = param.numel()
            param.data = params[idx:idx + param_numel].view_as(param).data
            idx += param_numel
        
    def forward(self, image_patches: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            image_patches (torch.Tensor): Batch of image patches
                (B, nb patches, C, patch_size, patch_size).

        Raises:
            ValueError: If the input tensor if of incorrect dimension
            ValueError: If the input tensor if of incorrect shape

        Returns:
            torch.Tensor: Predictions of the module. 
        """
        # Check that the input tensor is of the correct dimension and shape
        if image_patches.dim() != 5:
            raise ValueError(
                "Input tensor is of incorrect shape. "
                f"Expected 5D tensor but got {image_patches.dim()}D tensor."
            )
        elif image_patches.shape[2:] != (self._nb_channels,
                                         self._patch_size,
                                         self._patch_size):
            raise ValueError(
                "Input tensor is of incorrect shape. Expected tensor of shape "
                + str((
                    image_patches.shape[0],
                    image_patches.shape[1],
                    self._nb_channels,
                    self._patch_size,
                    self._patch_size))
                + f" but got {tuple(image_patches.shape)}."
            )
        
        # Flatten the input tensor
        patches_flattened = image_patches.view(
            image_patches.size(0),
            image_patches.size(1),
            -1,
        )
        
        return self._mlp(patches_flattened)


if __name__ == "__main__":
    pass
