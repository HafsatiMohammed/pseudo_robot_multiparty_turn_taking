import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual projection if channels don't match
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: [B, in_channels, seq_len]
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        out = out + residual
        out = F.relu(out)

        return out


class FrameBranchTCN(nn.Module):
    """TCN-based frame branch for temporal feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels_list: List[int],
        kernel_size: int = 3,
        dropout: float = 0.3,
        output_dim: int = 128,
    ):
        """
        Args:
            in_channels: Number of input channels (7)
            out_channels_list: List of output channels for each TCN block
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            output_dim: Final embedding dimension
        """
        super().__init__()

        layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(out_channels_list):
            dilation = 2 ** i  # Exponential dilation for larger receptive field
            layers.append(
                TCNBlock(
                    prev_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            prev_channels = out_channels

        self.tcn_blocks = nn.Sequential(*layers)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(prev_channels, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        x = x.transpose(1, 2)  # [B, in_channels, seq_len]

        x = self.tcn_blocks(x)  # [B, out_channels, seq_len]

        x = self.global_pool(x)  # [B, out_channels, 1]
        x = x.squeeze(-1)  # [B, out_channels]

        x = self.projection(x)  # [B, output_dim]
        x = self.dropout(x)

        return x


class FrameBranchGRU(nn.Module):
    """GRU-based frame branch for temporal feature extraction."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 128,
        bidirectional: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels (7)
            hidden_dim: Hidden dimension of GRU
            num_layers: Number of GRU layers
            dropout: Dropout rate
            output_dim: Final embedding dimension
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        gru_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.projection = nn.Linear(gru_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        _, hidden = self.gru(x)  # hidden: [num_layers*num_directions, B, hidden_dim]

        # Take last layer's output
        if self.gru.bidirectional:
            hidden = hidden[-2:]  # Last 2 directions
            hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # [B, 2*hidden_dim]
        else:
            hidden = hidden[-1]  # [B, hidden_dim]

        x = self.projection(hidden)  # [B, output_dim]
        x = self.dropout(x)

        return x


class ScalarBranch(nn.Module):
    """MLP for scalar feature processing."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
    ):
        """
        Args:
            in_dim: Input dimension (6)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, in_dim]
        x = self.mlp(x)  # [B, output_dim]
        return x


class FusionModule(nn.Module):
    """Fusion MLP to combine frame and scalar embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_dim: Input dimension (frame_dim + scalar_dim)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (3 for WAIT/BACKCHANNEL/START_SPEAKING)
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, in_dim]
        logits = self.mlp(x)  # [B, output_dim]
        return logits
