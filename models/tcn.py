# models/tcn.py

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """
    Remove extra elements introduced by padding in causal convolutions.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCN(nn.Module):
    """
    Generic TCN for sequence-to-sequence regression.

    Inputs:
        x: [batch, seq_len, input_dim]

    Outputs:
        y: [batch, seq_len]   (regression targets per time step)
        repr: [batch, repr_dim] (global representation, e.g. last time step hidden)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: List[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.output_head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    @property
    def repr_dim(self) -> int:
        return self.network[-1].conv2.out_channels

    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len, input_dim]
        """
        # TCN expects [batch, channels, seq_len]
        x = x.transpose(1, 2)

        h = self.network(x)  # [batch, C, seq_len]

        # per-time-step regression
        out = self.output_head(h)  # [batch, 1, seq_len]
        out = out.squeeze(1)       # [batch, seq_len]

        # global representation: last time step of last layer
        repr_vec = h[:, :, -1]     # [batch, C]

        return out, repr_vec
