import torch.nn as nn
from src.transformer.layers import ResidualConnection

class DecoderBlock(nn.Module):
    """Defines a single block of the decoder in a Transformer model."""

    def __init__(self, self_attention, cross_attention, feed_forward, dropout):
        """Initializes the decoder block with attention mechanisms and feed-forward layers."""
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward

        self.res_connection_1 = ResidualConnection(dropout)
        self.res_connection_2 = ResidualConnection(dropout)
        self.res_connection_3 = ResidualConnection(dropout)

    def forward(self, x, encoder_x, encoder_mask, decoder_mask):
        """Applies self-attention, cross-attention, and feed-forward layers with residual connections."""
        x = self.res_connection_1(x, lambda x: self.self_attention(x, x, x, decoder_mask))
        x = self.res_connection_2(x, lambda x: self.cross_attention(x, encoder_x, encoder_x, encoder_mask))
        x = self.res_connection_3(x, self.feed_forward)
        return x
