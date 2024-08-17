import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """Projects the model's output to the vocabulary size and applies log-softmax."""

    def __init__(self, d_model, vocabulary_size):
        """Initializes the linear projection layer."""
        super().__init__()
        self.projection = nn.Linear(d_model, vocabulary_size)

    def forward(self, x):
        """Applies linear projection followed by log-softmax to the input tensor."""
        return torch.log_softmax(self.projection(x), dim=-1)
