import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    """Embeds input tokens into dense vectors and scales by the model dimension."""
    
    def __init__(self, d_model, vocabular_size):
        """Initializes the embedding layer with the specified vocabulary size and model dimension."""
        super().__init__()
        self.d_model = d_model
        self.vocabular_size = vocabular_size
        self.embedding = nn.Embedding(vocabular_size, d_model)

    def forward(self, x):
        """Embeds the input tokens and scales the embedding by the square root of the model dimension."""
        return self.embedding(x) * torch.sqrt(torch.tensor([self.d_model]))
