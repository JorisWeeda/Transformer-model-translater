import torch.nn as nn
from src.transformer.layers import NormalizationLayer

class Decoder(nn.Module):
    """Decoder module for a transformer model."""

    def __init__(self, layers):
        """Initializes the decoder with the given layers."""
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.norm = NormalizationLayer()

    def forward(self, x, encoder_x, encoder_mask, decoder_mask):
        """Passes input through the decoder layers and normalization."""
        for layer in self.layers:
            x = layer(x, encoder_x, encoder_mask, decoder_mask)

        return self.norm(x)
