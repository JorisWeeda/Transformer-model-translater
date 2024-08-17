import torch.nn as nn
from src.transformer.layers import ResidualConnection

class EncoderBlock(nn.Module):
    """Defines a single block of the encoder in a Transformer model."""

    def __init__(self, self_attention, feed_forward, dropout):
        """Initializes the encoder block with self-attention and feed-forward layers."""
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward

        self.res_connection_1 = ResidualConnection(dropout)
        self.res_connection_2 = ResidualConnection(dropout)

    def forward(self, x, encoder_mask):
        """Applies self-attention and feed-forward layers with residual connections."""
        x = self.res_connection_1(x, lambda x: self.self_attention(x, x, x, encoder_mask))
        x = self.res_connection_2(x, self.feed_forward)
        return x
