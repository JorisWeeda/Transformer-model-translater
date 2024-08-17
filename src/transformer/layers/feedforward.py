import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """Implements the position-wise feedforward network used in Transformers."""
    
    def __init__(self, d_model, d_ff, dropout):
        """Initializes the feedforward network with given dimensions and dropout."""
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Applies a two-layer feedforward network with ReLU activation and dropout."""
        x = torch.relu(self.linear_1(x))
        x = self.linear_2(self.dropout(x))
        return x
