# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class ConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        nb_layers,
        output_activation=F.relu,
        down_scaling=False,
        use_residual=True,
    ) -> None:
        
        super(ConvBlock, self).__init__()

        conv_layers = []
        
        # Create all the convolutional layers for the block
        for i in range(nb_layers):
            
            # The first layer of the block may be different if downscaling is required
            # (downscaling is done by using a stride of 2 in the convolutional layer,
            # not by using a pooling layer which induces a loss of information)
            if i == 0:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4 if down_scaling else 3,
                        stride=2 if down_scaling else 1,
                        padding=1,
                    )
                )
            # The other layers do not change the input dimensions
            else:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                    )
                )
        self._conv_layers = nn.ModuleList(conv_layers)
        self._output_activation = output_activation
        # self._output_activation = nn.ReLU()
        self._use_residual = use_residual and nb_layers > 2

    def forward(self, x):
        
        # Apply the convolutional layers
        for i in range(len(self._conv_layers)):
            
            # We store the tensor obtained after the first convolutional layer
            # to perform the residual connection
            if i == 1 and self._use_residual:
                x_at_start = x.clone()
            
            # Apply the convolutional layer
            x = self._conv_layers[i](x)
            
            # Apply the output activation function
            if i == len(self._conv_layers) - 1:
                x = self._output_activation(x)
            else:
                x = F.relu(x)

        # Add the residual connection to allow the gradient to flow through the network
        # more easily
        if self._use_residual:
            x += x_at_start
        
        return x

# class FiLM(nn.Module):
#     def __init__(self, in_channels, context_dim):
#         super(FiLM, self).__init__()
#         self.gamma_fc = nn.Linear(context_dim, in_channels)
#         self.beta_fc = nn.Linear(context_dim, in_channels)

#     def forward(self, x, context):
#         gamma = self.gamma_fc(context).unsqueeze(2).unsqueeze(3).expand_as(x)
#         beta = self.beta_fc(context).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return gamma * x + beta

            
class UNet1dEncoder(nn.Module):
    
    def __init__(self, channels_list, nb_layers_per_block=1):
        
        super(UNet1dEncoder, self).__init__()
        
        conv_blocks = []
        
        # Create all the convolutional blocks for the encoder
        for i in range(len(channels_list) - 1):
            
            in_channels, out_channels = channels_list[i:i+2]
            
            conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    nb_layers=nb_layers_per_block,
                    down_scaling=True
                )
            )
        self._conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        
        intermediate_states = []
        
        # Apply the convolutional blocks
        for i, conv in enumerate(self._conv_blocks):
            
            # Apply the convolutional block
            x = conv(x)
            
            # Store the intermediate tensors to concatenate them in the decoder
            if i < len(self._conv_blocks) - 1:
                intermediate_states.append(x.clone())
        
        return x, intermediate_states
    

class UNet1dDecoder(nn.Module):
    
    def __init__(self, channels_list, nb_layers_per_block):
        
        super(UNet1dDecoder, self).__init__()
        
        conv_blocks = []
        
        nb_layers = len(channels_list) - 1
        
        # Create all the convolutional blocks for the decoder
        for i in range(nb_layers):
            
            in_channels = channels_list[i] * (2 if i > 0 else 1)
            out_channels = channels_list[i + 1]
            
            # The last layer reduces the number of channels
            if i == nb_layers - 1:
                conv_blocks.append(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        nb_layers=1,
                        output_activation=lambda x: x,
                    )
                )
            else:
                conv_blocks.append(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        nb_layers=nb_layers_per_block,
                        output_activation=F.relu,
                    )
                )
        self._conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x, skip_connections):
        
        # Apply the convolutional blocks
        for i, conv in enumerate(self._conv_blocks):
            
            # Skip connection concatenation
            if i > 0:
                x = torch.cat([x, skip_connections[-i]], axis=1)
            
            # Upsample and apply the convolutional block
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = conv(x)
        
        return x

class FiLMedUNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels_list: list = [16, 32, 64],
        nb_layers_per_block_encoder: int = 3,
        nb_layers_bridge: int = 3,
        nb_layers_per_block_decoder: int = 3,
        output_logits: bool = True,
    ):
        super(FiLMedUNet1d, self).__init__()
        
        # Create the encoder, bridge and decoder
        down_channels = [in_channels] + channels_list
        self._encoder = UNet1dEncoder(
            channels_list=down_channels,
            nb_layers_per_block=nb_layers_per_block_encoder,
        )
        self._bridge  = ConvBlock(
            in_channels=channels_list[-1],
            out_channels=channels_list[-1],
            nb_layers=nb_layers_bridge,
        )
        up_channels = channels_list[::-1] + [out_channels]
        self._decoder = UNet1dDecoder(
            channels_list=up_channels,
            nb_layers_per_block=nb_layers_per_block_decoder,
        )
        
        # The sigmoid activation is to be applied in inference mode ; in training mode,
        # it is usually included in the loss function to ensure numerical stability
        self._output_activation = nn.Identity() if output_logits else nn.Sigmoid()
        
    def forward(self, x):
        # Encode the input and get the intermediate states
        x, intermediate_states = self._encoder(x)
        
        # Pass the encoded tensor through the bridge
        x = self._bridge(x)
        
        # Decode the tensor using the intermediate states from the encoder
        # to perform the skip connections
        x = self._decoder(x, intermediate_states)
        
        x = self._output_activation(x)
        
        return x
    

if __name__ == "__main__":
    
    config = {
        "in_channels": 3,
        "out_channels": 1,
        "channels_list": [16, 32, 64],  # Number of channels after each block
        "nb_layers_per_block_encoder": 3,
        "nb_layers_bridge": 3,
        "nb_layers_per_block_decoder": 3,
        "output_logits": True,  # Whether to output logits or probabilities
    }
    
    # Display the model architecture
    torchinfo.summary(
        FiLMedUNet1d(**config),
        input_size=(1, 3, 120),
    )
