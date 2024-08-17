import torch.nn as nn
from src.transformer.layers import NormalizationLayer

class Encoder(nn.Module):
    """Encoder module for a transformer model."""

    def __init__(self, layers):
        """Initializes the encoder with the given layers."""
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.norm = NormalizationLayer()

    def forward(self, x, encoder_mask):
        """Passes input through the encoder layers and normalization."""
        for layer in self.layers:
            x = layer(x, encoder_mask)

        return self.norm(x)
