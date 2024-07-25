# Standard libraries
from typing import Optional

# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class ResamplingLayer1d(nn.Module):
    """
    Downsamples or upsamples the input tensor by a factor of 2, or does nothing
    if the number of input and output channels is the same. The resampling
    is done by using a learnable convolutional layer to avoid information loss.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Raises:
            ValueError: If the number of input and/or output channels is invalid.
        """
        super(ResamplingLayer1d, self).__init__()
        
        # Define the resampling layer
        self._resampling_layer = lambda x: x
        if in_channels != out_channels:
            # Downsampling
            if out_channels == in_channels * 2:
                # We can use a max pooling layer instead of a convolutional layer
                # to improve the computational efficiency (but it leads to a loss
                # of expressiveness)
                self._resampling_layer = nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                )
            # Upsampling
            elif out_channels == in_channels / 2:
                # We can use an upsampling layer instead of a convolutional layer
                # to improve the computational efficiency (but it leads to a loss
                # of expressiveness)
                self._resampling_layer = nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                )
            else:
                raise ValueError(
                    f"Invalid number of channels: {in_channels} -> {out_channels}"
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self._resampling_layer(x)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer to conditionally transform
    an input tensor based on a context vector.
    """
    def __init__(self, in_channels: int, context_dim: int) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            context_dim (int): Dimension of the context vector.
        """
        super(FiLM, self).__init__()
        
        # Fully connected layers to compute scale and shift factors
        # from a context vector
        self._gamma_fc = nn.Linear(context_dim, in_channels)
        self._beta_fc = nn.Linear(context_dim, in_channels)
        
        # Initialize the parameters so that the FiLM layer is initially
        # the identity function
        self._gamma_fc.weight.data.fill_(0)
        self._gamma_fc.bias.data.fill_(1)
        self._beta_fc.weight.data.fill_(0)
        self._beta_fc.bias.data.fill_(0)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor. Shape [B, C, L]. B is the number of lines,
                C the number of channels and L the length of the line.
            context (torch.Tensor): Context vector. Shape [1, D]. The same context
                vector is used for all the lines. D is the dimension of the context
                space.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Compute the scale and shift factors
        gamma = self._gamma_fc(context).unsqueeze(2)  # [1, C, 1]
        beta = self._beta_fc(context).unsqueeze(2)  # [1, C, 1]
        
        # Transform the input tensor (broadcasting)
        return gamma * x + beta  # [B, C, L]


class ConvBlock1d(nn.Module):
    """
    A series of convolutional layers with batch normalization and ReLU activation.
    If the number of input and output channels is different, the first layer of
    the block is used to adjust the number of channels, and the other layers
    keep the number of channels the same. A residual connection may optionally
    be added if the number of layers (excluding the first layer if it is used to adjust
    the number of channels) is greater than 2.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nb_layers: int,
        use_residual: bool = True,
        film_dim: Optional[int] = None,
    ) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            nb_layers (int): Number of convolutional layers in the block.
            use_residual (bool, optional): Whether to use a residual connection
                within the block. Only applies if the number of layers is greater
                is greater than 3, or if the number of layers is greater than 2 and
                the number of input and output channels is the same. Defaults to True.
            film (bool, optional): Whether to use Feature-wise Linear Modulation
                (FiLM). Defaults to False.
        """
        super(ConvBlock1d, self).__init__()
        
        # Create a first layer to adjust the number of channels if needed
        self._first_conv = None
        if in_channels != out_channels:
            self._first_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),
                nn.BatchNorm1d(out_channels),
                FiLM(out_channels, film_dim) if nb_layers == 1 and film_dim is not None\
                    else nn.Identity(),
                nn.ReLU(),
            )
            nb_layers -= 1
        
        # Create the other convolutional layers
        # (do not change the number of channels)
        conv_layers = []
        for i in range(nb_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                    ),
                    nn.BatchNorm1d(out_channels),
                    # FiLM layer added only to the last layer
                    FiLM(out_channels, film_dim) if i == nb_layers - 1 and film_dim\
                        is not None else nn.Identity(),
                    nn.ReLU() if i < nb_layers - 1 else nn.Identity(),
                )
            )
        self._conv_layers = nn.Sequential(*conv_layers)
        
        # Eventually use a residual connection within the block
        self._use_residual = use_residual and nb_layers > 2
        
        # Eventually use FiLM to condition the convolutional block
        self._use_film = isinstance(film_dim, int)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            context (torch.Tensor, optional): Context vector for FiLM. Defaults to
                None.

        Returns:
            torch.Tensor: Output tensor.
        
        Raises:
            ValueError: If FiLM is used but no context vector is provided.
        """
        # First layer to adjust the number of channels if needed
        if self._first_conv is not None:
            if len(self._conv_layers) == 0 and self._use_film and context is not None:
                for mod in self._first_conv:
                    x = mod(x, context) if isinstance(mod, FiLM) else mod(x)
            elif len(self._conv_layers) > 0 or not self._use_film:
                x = self._first_conv(x)
            else:
                raise ValueError("FiLM requires a context vector")
        
        # Store the initial tensor to perform the residual connection if needed
        if self._use_residual:
            x_at_start = x.clone()
            
        # Apply the convolutional layers
        if self._use_film and context is not None:
            for conv_layer in self._conv_layers:
                for mod in conv_layer:
                    x = mod(x, context) if isinstance(mod, FiLM) else mod(x)
        elif not self._use_film:
            x = self._conv_layers(x)
        else:
            raise ValueError("FiLM requires a context vector")
        
        # Perform the residual connection if needed
        if self._use_residual:
            x += x_at_start
        
        x = F.relu(x)
        
        return x


class UNetEncoder1d(nn.Module):
    """
    The encoder of the U-Net 1D architecture. It consists of a series of
    convolutional blocks followed by downsampling layers.
    """
    def __init__(
        self,
        in_channels: int,
        channels_list: list,
        nb_layers_per_block: int = 1,
        film_dim: Optional[int] = None,
    ) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            channels_list (list): List of the number of channels at each scale.
            nb_layers_per_block (int, optional): Number of convolutional layers
                in each block. Defaults to 1.
            film_dim (int, optional): Dimension of the context vector for FiLM.
                Defaults to None.
        """
        super(UNetEncoder1d, self).__init__()
        
        # Create all the convolutional blocks and downsampling layers
        self._layers = nn.ModuleList()
        for i in range(0, len(channels_list) - 1):
            # First block has a different number of input channels and
            # other blocks the same number of input and output channels
            in_channels_i = in_channels if i == 0 else channels_list[i]
            # Convolutional block
            self._layers.append(
                ConvBlock1d(
                    in_channels=in_channels_i,
                    out_channels=channels_list[i],
                    nb_layers=nb_layers_per_block,
                    film_dim=film_dim,
                )
            )
            # Downsampling layer
            self._layers.append(
                ResamplingLayer1d(
                    in_channels=channels_list[i],
                    out_channels=channels_list[i+1],
                )
            )
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            context (torch.Tensor, optional): Context vector for FiLM. Defaults to
                None.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Output tensor and list of
                intermediate tensors to be sent to the decoder.
        """
        # List to store the intermediate tensors to communicate to the decoder
        intermediate_states = []
        
        for i in range(0, len(self._layers), 2):
        
            # Apply the convolutional block
            x = self._layers[i](x, context)
            
            # Store the intermediate tensors to concatenate them in the decoder
            intermediate_states.append(x.clone())
            
            # Apply the downsampling layer
            x = self._layers[i+1](x)
            
        return x, intermediate_states
    

class UNetDecoder1d(nn.Module):
    """
    The decoder of the U-Net 1D architecture. It consists of a series of
    upsampling layers followed by convolutional blocks.
    """
    def __init__(
        self,
        out_channels: int,
        channels_list: list,
        nb_layers_per_block: int = 1,
    ) -> None:
        """Constructor.

        Args:
            out_channels (int): Number of output channels.
            channels_list (list): List of the number of channels at each scale.
            nb_layers_per_block (int, optional): Number of convolutional layers
                in each block. Defaults to 1.
        """
        super(UNetDecoder1d, self).__init__()
        
        # Create all the convolutional blocks and upsampling layers
        self._layers = nn.ModuleList()
        for i in range(0, len(channels_list) - 1):
            # Upsampling layer
            self._layers.append(
                ResamplingLayer1d(
                    in_channels=channels_list[i],
                    out_channels=channels_list[i+1],
                )
            )
            # Convolutional block
            # (different number of input channels and output channels)
            self._layers.append(
                ConvBlock1d(
                    in_channels=channels_list[i],
                    out_channels=channels_list[i+1],
                    nb_layers=nb_layers_per_block,
                )
            )
        
        # Last layer to reduce the number of channels to the number of output channels
        self._output_layer = nn.Conv1d(
            in_channels=channels_list[-1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        intermediate_states: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            intermediate_states (list[torch.Tensor]): List of intermediate tensors
                from the encoder.

        Returns:
            torch.Tensor: Output tensor.
        """
        for i in range(0, len(self._layers), 2):
            
            # Apply the upsampling layer
            x = self._layers[i](x)
            
            # Concatenate the intermediate tensor from the encoder
            x = torch.cat([x, intermediate_states[-(i//2+1)]], axis=1)
        
            # Apply the convolutional block
            x = self._layers[i+1](x)
        
        x = self._output_layer(x)
            
        return x


class UNet1d(nn.Module):
    """
    The U-Net 1D architecture. It consists of an encoder, a bridge and a decoder.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels_list: list = [16, 32, 64],
        nb_layers_per_block_encoder: int = 3,
        nb_layers_bridge: int = 3,
        nb_layers_per_block_decoder: int = 3,
        film_dim: Optional[int] = None,
        output_logits: bool = True,
    ) -> None:
        """Constructor.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            channels_list (list, optional): List of the number of channels at each
                scale. Defaults to [16, 32, 64].
            nb_layers_per_block_encoder (int, optional): Number of convolutional
                layers in each block of the encoder. Defaults to 3.
            nb_layers_bridge (int, optional): Number of convolutional layers in the
                bridge between the encoder and the decoder. Defaults to 3.
            nb_layers_per_block_decoder (int, optional): Number of convolutional
                layers in each block of the decoder. Defaults to 3.
            film_dim (int, optional): Dimension of the context vector for FiLM.
                Defaults to None.
            output_logits (bool, optional): Whether to output logits or probabilities
                (sigmoid activation). Defaults to True.
        """
        super(UNet1d, self).__init__()
        
        # Used to check the width of the input tensor
        self._divisor = 2**(len(channels_list)-1)
        
        # Create the encoder, bridge and decoder
        self._encoder = UNetEncoder1d(
            in_channels=in_channels,
            channels_list=channels_list,
            nb_layers_per_block=nb_layers_per_block_encoder,
            film_dim=film_dim,
        )
        self._bridge  = ConvBlock1d(
            in_channels=channels_list[-1],
            out_channels=channels_list[-1],
            nb_layers=nb_layers_bridge,
            film_dim=film_dim,
        )
        self._decoder = UNetDecoder1d(
            out_channels=out_channels,
            channels_list=channels_list[::-1],
            nb_layers_per_block=nb_layers_per_block_decoder,
        ) 
        
        # The sigmoid activation is to be applied in inference mode ; in training mode,
        # it is usually included in the loss function to ensure numerical stability
        self._output_activation = nn.Identity() if output_logits else nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            context (torch.Tensor, optional): Context vector for FiLM. Defaults to
                None.

        Raises:
            ValueError: If the width of the input tensor is not divisible by the scale
            factor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Check the width of the input tensor
        if x.size(-1) % self._divisor != 0:
            raise ValueError(
                f"Invalid width of the input tensor: {x.size(-1)} "
                f"(should be a multiple of {self._divisor})"
            )
        
        # Encode the input and get the intermediate states
        x, intermediate_states = self._encoder(x, context)
        
        # Pass the encoded tensor through the bridge
        x = self._bridge(x, context)
        
        # Decode the tensor using the intermediate states from the encoder
        # to perform the skip connections
        x = self._decoder(x, intermediate_states)
        
        # Apply the activation function
        x = self._output_activation(x)
        
        return x


if __name__ == "__main__":
    
    config = {
        "in_channels": 3,
        "out_channels": 1,
        "channels_list": [16, 32, 64, 128],  # Number of channels after each block
        "nb_layers_per_block_encoder": 3,
        "nb_layers_bridge": 3,
        "nb_layers_per_block_decoder": 3,
        "film_dim": 12,  # Dimension of the context vector for FiLM
        "output_logits": True,  # Whether to output logits or probabilities
    }
    
    # Display the model architecture
    torchinfo.summary(
        UNet1d(**config),
        input_size=[(10, 3, 120), (1, 12)],
        device="cpu",
    )
    
    # x = torch.randn(1, 1, 12)
    # film = FiLM(1, 10)
    # y = film(x, torch.randn(1, 10))
    
    # model = UNet1d(**config)
    # x = torch.randn(2, 3, 120)
    # y = model(x, torch.randn(1, 12))
    
    # Export the model to ONNX
    # model = UNet1d(**config)
    # x = torch.randn(1, 3, 120)
    # torch.onnx.export(
    #     model,
    #     x,
    #     "model.onnx",
    #     verbose=False,
    #     training=torch.onnx.TrainingMode.TRAINING,
    # )
