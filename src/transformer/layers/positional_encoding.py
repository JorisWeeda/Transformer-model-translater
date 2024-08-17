import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Injects positional information into input embeddings using sine and cosine functions."""

    PE_CONSTANT = 10000.0

    def __init__(self, d_model, sequence_length, dropout):
        """Initializes the positional encoding matrix and dropout layer."""
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor([self.PE_CONSTANT])) / d_model))

        self.pos_encoding = torch.zeros(sequence_length, d_model)
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = self.pos_encoding.unsqueeze(0)

        self.register_buffer('pe', self.pos_encoding)

    def forward(self, x):
        """Adds positional encoding to the input tensor and applies dropout."""
        x = x + (self.pos_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
