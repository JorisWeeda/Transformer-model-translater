import torch.nn as nn
from . import NormalizationLayer

class ResidualConnection(nn.Module):
    """Applies a residual connection followed by normalization to the input."""

    def __init__(self, dropout):
        """Initializes the dropout layer and normalization layer."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = NormalizationLayer()

    def forward(self, x, sub_layer):
        """Applies the residual connection and normalization to the sub-layer output."""
        return x + self.dropout(sub_layer(self.normalization(x)))
