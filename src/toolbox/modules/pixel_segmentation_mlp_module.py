import torch
import torch.nn as nn
import torchinfo
import torch.nn.functional as F


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
        output_size: int = 1,
        hidden_dims: list[int] = [64, 32, 16],
        output_logits: bool = True,
    ) -> None:
        """Constructor of the class.

        Args:
            patch_size (int, optional): Side length of the square patch. Defaults to 5.
            nb_channels (int, optional): Number of channels in the input tensor.
                Defaults to 3.
            output_size (int, optional): Size of the output tensor. Defaults to 1.
            hidden_dims (list[int], optional): Number of hidden units in each layer of
                the MLP. Defaults to [128, 64, 32].
            output_logits (bool, optional): Whether to output logits or probabilities.
                Defaults to True.
        """
        super(PixelSegmentationMLP, self).__init__()
        
        self._hidden_dims = hidden_dims
        self._patch_size = patch_size
        self._nb_channels = nb_channels
        self._output_size = output_size
        self._output_logits = output_logits
        
        self._nb_parameters = self.get_nb_parameters_mlp(
            input_size=patch_size ** 2 * nb_channels,
            hidden_dims=hidden_dims,
            output_size=output_size,
        )
        
    @staticmethod
    def get_nb_parameters_mlp(
        input_size: int,
        hidden_dims: list[int],
        output_size: int,
    ) -> int:
        """Compute the number of parameters in a MLP.

        Args:
            input_size (int): Size of the input tensor.
            hidden_dims (list[int]): Number of hidden units in each layer of the MLP.
            output_size (int): Size of the output tensor.

        Returns:
            int: Number of parameters in the MLP.
        """
        nb_parameters = 0
        in_features = input_size
        
        for hidden_dim in hidden_dims:
            
            nb_parameters += in_features * hidden_dim  # Weights
            nb_parameters += hidden_dim  # Biases
            
            in_features = hidden_dim
        
        nb_parameters += in_features * output_size  # Final layer weights
        nb_parameters += output_size  # Final layer biases
        
        return nb_parameters
        
    @property
    def nb_parameters(self) -> int:
        """Return the number of parameters of the model.

        Returns:
            int: Number of parameters of the model.
        """
        return self._nb_parameters
    
    def forward_mlp(self, x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP. One prediction per patch.

        Args:
            x (torch.Tensor): Input tensor of shape (B, nb patches, nb features).
            parameters (torch.Tensor): Parameters of the model.

        Returns:
            torch.Tensor: Predictions of the module (B, nb patches, output_size).
        """
        # Ensure the tensor is a 1D tensor
        parameters = parameters.flatten()
        
        in_features = self._patch_size ** 2 * self._nb_channels
        start_idx = 0
        
        for hidden_dim in self._hidden_dims:
            end_idx = start_idx + in_features * hidden_dim
            weight = parameters[start_idx:end_idx].reshape(hidden_dim,
                                                           in_features)
            start_idx = end_idx
            end_idx = start_idx + hidden_dim
            bias = parameters[start_idx:end_idx]
            start_idx = end_idx

            x = F.linear(x, weight, bias)
            x = F.relu(x)
            in_features = hidden_dim
        
        # Final output layer
        end_idx = start_idx + in_features * self._output_size
        weight = parameters[start_idx:end_idx].reshape(self._output_size,
                                                       in_features)
        start_idx = end_idx
        bias = parameters[start_idx:start_idx + self._output_size]

        x = F.linear(x, weight, bias)
        
        if not self._output_logits:
            x = torch.sigmoid(x)
        
        return x

    def forward(
        self,
        image_patches: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            image_patches (torch.Tensor): Batch of image patches
                (B, nb patches, C, patch_size, patch_size).
            parameters (torch.Tensor): Parameters of the model.

        Raises:
            ValueError: If the input tensor if of incorrect dimension
            ValueError: If the input tensor if of incorrect shape
            ValueError: If the number of tensor values does not match the number of
                model parameters

        Returns:
            torch.Tensor: Predictions of the module (B, nb patches, output_size).
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
        
        # Ensure the parameter tensor has the correct number of values
        if parameters.numel() != self._nb_parameters:
            raise ValueError(
                "The number of tensor values must match the number of model parameters"
            )
        
        return self.forward_mlp(patches_flattened, parameters)


if __name__ == "__main__":
    
    # Instantiate the model
    mlp = PixelSegmentationMLP()
    
    # Set the parameters of the model
    params = torch.randn(mlp.nb_parameters)
    
    # Create a random input tensor with appropriate shape
    input_tensor = torch.randn(2, 10, 3, 5, 5)
    print("Input shape: ", input_tensor.shape)
    
    output = mlp.forward(input_tensor, params)
    print("Output shape: ", output.shape)
