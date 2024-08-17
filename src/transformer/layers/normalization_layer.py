import torch
import torch.nn as nn

class NormalizationLayer(nn.Module):
    """Applies layer normalization to the input tensor."""
    
    EPSILON = 1e-5  # Adjusted for clarity and common practice

    def __init__(self):
        """Initializes the normalization layer with learnable scale (alpha) and shift (bias) parameters."""
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Normalizes the input tensor by subtracting the mean and dividing by the standard deviation."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.EPSILON) + self.bias
