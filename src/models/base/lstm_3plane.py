"""
LSTM-based model for 3-plane flow sequence prediction.
Processes 12-channel input (3 planes × 4 fields: u,v,w,p) for spatio-temporal prediction.
"""

import einops
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for spatial-temporal modeling.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Args:
            input_dim: Number of channels of input tensor
            hidden_dim: Number of channels of hidden state
            kernel_size: Size of the convolutional kernel (int or tuple)
            bias: Whether to add bias
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Handle kernel_size - convert to int if needed
        # Check if it's a list/tuple/ListConfig by trying to access index
        try:
            # If it's iterable and not a string, get the first element
            if hasattr(kernel_size, "__iter__") and not isinstance(kernel_size, str):
                kernel_size = kernel_size[0]
        except (TypeError, IndexError):
            pass
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.bias = bias

        # Convolutional gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        """
        Args:
            input_tensor: (B, C, H, W)
            cur_state: tuple of (h_cur, c_cur) each (B, hidden_dim, H, W)
        Returns:
            h_next, c_next: next hidden and cell states
        """
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)  # cell gate

        # Compute next cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize hidden and cell states with zeros.
        """
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        kernel_sizes,
        num_layers,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        """
        Args:
            input_dim: Number of channels in input
            hidden_dims: List of hidden dimensions for each layer
            kernel_sizes: List of kernel sizes for each layer
            num_layers: Number of LSTM layers
            batch_first: Whether batch is first dimension
            bias: Whether to use bias
            return_all_layers: Return outputs from all layers
        """
        super().__init__()

        self.input_dim = input_dim
        # Convert ListConfig/list/tuple to regular list if needed
        if hasattr(hidden_dims, "__iter__") and not isinstance(hidden_dims, str):
            self.hidden_dims = list(hidden_dims)
        else:
            self.hidden_dims = [hidden_dims] * num_layers

        if hasattr(kernel_sizes, "__iter__") and not isinstance(kernel_sizes, str):
            self.kernel_sizes = list(kernel_sizes)
        else:
            self.kernel_sizes = [kernel_sizes] * num_layers

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_sizes[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Args:
            input_tensor: (B, T, C, H, W) or (T, B, C, H, W)
            hidden_state: Initial hidden states
        Returns:
            layer_output_list: List of outputs from each layer
            last_state_list: List of last states from each layer
        """
        if not self.batch_first:
            # Convert (T, B, C, H, W) to (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        B, T, _, H, W = input_tensor.size()

        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=B, image_size=(H, W))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(T):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)  # (B, T, hidden_dim, H, W)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class LSTM3Plane(nn.Module):
    """
    LSTM-based encoder-decoder for 3-plane flow prediction.
    Processes 12-channel input (3 planes × 4 fields) and predicts future states.
    """

    def __init__(
        self,
        input_shape=None,
        sequence_length=5,
        prediction_horizon=1,
        num_channels=12,  # 3 planes × 4 fields = 12 channels
        hidden_dims=None,
        kernel_sizes=None,
        num_layers=3,
        bias=True,
        **kwargs,
    ):
        """
        Args:
            input_shape: Spatial dimensions [H, W]
            sequence_length: Number of input timesteps
            prediction_horizon: Number of timesteps to predict
            num_channels: Number of input channels (12 for 3-plane)
            hidden_dims: List of hidden dimensions for each LSTM layer
            kernel_sizes: List of kernel sizes for each LSTM layer
            num_layers: Number of LSTM layers
            bias: Whether to use bias in ConvLSTM
        """
        super().__init__()

        # Set default values
        if input_shape is None:
            input_shape = [128, 128]
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        # Convert to regular Python lists to handle ListConfig from Hydra
        if hasattr(input_shape, "__iter__") and not isinstance(input_shape, str):
            self.input_shape = list(input_shape)
        else:
            self.input_shape = input_shape

        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.num_channels = num_channels

        if hasattr(hidden_dims, "__iter__") and not isinstance(hidden_dims, str):
            self.hidden_dims = list(hidden_dims)
        else:
            self.hidden_dims = hidden_dims

        if hasattr(kernel_sizes, "__iter__") and not isinstance(kernel_sizes, str):
            self.kernel_sizes = list(kernel_sizes)
        else:
            self.kernel_sizes = kernel_sizes

        self.num_layers = num_layers

        # Input projection - reduce channels before LSTM
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                num_channels, self.hidden_dims[0], kernel_size=3, padding=1, bias=bias
            ),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder LSTM
        self.encoder = ConvLSTM(
            input_dim=self.hidden_dims[0],
            hidden_dims=self.hidden_dims,
            kernel_sizes=self.kernel_sizes,
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
            return_all_layers=True,
        )

        # Decoder LSTM (processes encoded representation)
        self.decoder = ConvLSTM(
            input_dim=self.hidden_dims[-1],
            hidden_dims=list(reversed(self.hidden_dims)),
            kernel_sizes=list(reversed(self.kernel_sizes)),
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
            return_all_layers=False,
        )

        # Output projection - map from hidden dim back to channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(
                self.hidden_dims[0],
                self.hidden_dims[0],
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dims[0], num_channels, kernel_size=1, bias=bias),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) input sequence where C = 12
        Returns:
            output: (B, T_pred, C, H, W) predicted sequence or (B, C, H, W) if T_pred=1
        """
        # Handle flattened input
        if len(x.shape) == 4:
            BT, C, H, W = x.shape
            B = BT // self.sequence_length
            T = self.sequence_length
            x = einops.rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T)
        else:
            B, T, C, H, W = x.shape

        assert self.sequence_length == T, (
            f"Expected sequence length {self.sequence_length}, got {T}"
        )
        assert self.num_channels == C, f"Expected {self.num_channels} channels, got {C}"
        assert tuple(self.input_shape) == (H, W), (
            f"Expected shape {self.input_shape}, got {(H, W)}"
        )

        # Apply input projection to each timestep
        x_projected = []
        for t in range(T):
            x_t = self.input_conv(x[:, t, :, :, :])  # (B, hidden_dims[0], H, W)
            x_projected.append(x_t)
        x_projected = torch.stack(x_projected, dim=1)  # (B, T, hidden_dims[0], H, W)

        # Encode sequence
        _, encoder_states = self.encoder(x_projected)

        # Prepare decoder input - use last encoded state
        # decoder_input: (B, 1, hidden_dims[-1], H, W)
        decoder_input = encoder_states[-1][0].unsqueeze(
            1
        )  # Use hidden state from last encoder layer

        # Decode for prediction_horizon steps
        predictions = []
        decoder_state = None

        for _ in range(self.prediction_horizon):
            # Decode one step
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)

            # decoder_output is a list, take the last layer's output
            decoded = decoder_output[-1][:, -1, :, :, :]  # (B, hidden_dims[0], H, W)

            # Project to output channels
            pred = self.output_conv(decoded)  # (B, C, H, W)
            predictions.append(pred)

            # For next iteration, use current prediction (autoregressive)
            if self.prediction_horizon > 1:
                # Project prediction back to decoder input space
                decoder_input = self.input_conv(pred).unsqueeze(1)

        # Stack predictions
        output = torch.stack(predictions, dim=1)  # (B, T_pred, C, H, W)

        # If single-step prediction, remove time dimension
        if self.prediction_horizon == 1:
            output = output.squeeze(1)  # (B, C, H, W)

        return output


def LSTM3PlaneAuto(**kwargs):
    """Factory function for LSTM 3-plane model."""
    return LSTM3Plane(**kwargs)
