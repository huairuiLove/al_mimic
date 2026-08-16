"""Package-local temporal adapter for supervised MDS-ED training.

This is a residual convolutional temporal encoder, not a full S4 reproduction.
It preserves the published four-layer waveform/tabular fusion contract while
keeping the training entry point independent of external source checkouts.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dimension)
        self.depthwise = nn.Conv1d(dimension, dimension, kernel_size=9, padding=4, groups=dimension)
        self.pointwise = nn.Conv1d(dimension, dimension * 2, kernel_size=1)
        self.output = nn.Conv1d(dimension, dimension, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(self.norm(values))
        values = self.pointwise(values)
        values, gate = values.chunk(2, dim=1)
        values = self.output(self.activation(values) * torch.sigmoid(gate))
        return residual + self.dropout(values)


class NativeMdsEdTemporalAdapter(nn.Module):
    """Four-layer waveform encoder fused with a three-layer tabular MLP."""

    backend_name = "native_temporal_adapter"

    def __init__(
        self,
        *,
        output_size: int,
        continuous_dim: int,
        category_sizes: tuple[int, ...],
        ecg_channels: int = 12,
        model_dim: int = 512,
        temporal_layers: int = 4,
        tabular_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_size < 1 or continuous_dim < 0 or model_dim < 1 or temporal_layers < 1:
            raise ValueError("invalid native MDS-ED model dimensions")
        self.waveform_input = nn.Conv1d(ecg_channels, model_dim, kernel_size=9, stride=2, padding=4)
        self.temporal_blocks = nn.Sequential(
            *(ResidualTemporalBlock(model_dim, dropout) for _ in range(temporal_layers))
        )
        self.waveform_norm = nn.LayerNorm(model_dim)
        self.embeddings = nn.ModuleList(nn.Embedding(size, 16) for size in category_sizes)
        tabular_input = continuous_dim + 16 * len(category_sizes)
        if tabular_input:
            self.tabular = nn.Sequential(
                nn.Linear(tabular_input, tabular_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(tabular_dim, tabular_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(tabular_dim, tabular_dim),
                nn.ReLU(),
            )
            fusion_dim = model_dim + tabular_dim
        else:
            self.tabular = None
            fusion_dim = model_dim
        self.classifier = nn.Linear(fusion_dim, output_size)

    def forward(
        self,
        ecg: torch.Tensor,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        if ecg.ndim != 3:
            raise ValueError(f"ECG batch must have shape [batch, samples, channels], got {ecg.shape}")
        waveform = self.waveform_input(ecg.transpose(1, 2))
        waveform = self.temporal_blocks(waveform).mean(dim=-1)
        waveform = self.waveform_norm(waveform)
        if self.tabular is None:
            fused = waveform
        else:
            pieces = [continuous]
            pieces.extend(embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings))
            fused = torch.cat((waveform, self.tabular(torch.cat(pieces, dim=1))), dim=1)
        return self.classifier(fused)
