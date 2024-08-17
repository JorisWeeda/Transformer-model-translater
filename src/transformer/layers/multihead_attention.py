import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    """Implements the Multihead Attention mechanism used in Transformers."""
    
    MASK_VALUE = -1e9

    def __init__(self, d_model, h, dropout):
        """Initializes the multihead attention with model dimensions, heads, and dropout."""
        super().__init__()
        
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_scores = None

    @classmethod
    def from_params(cls, d_model, h, dropout):
        """Creates an instance of MultiheadAttention with dimensionality checks."""
        assert d_model % h == 0, "d_model must be divisible by h"
        return cls(d_model, h, dropout)

    def forward(self, query, key, value, mask=None):
        """Performs the forward pass of multihead attention."""
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)

        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)

        x, attention_scores = self._attention(query, key, value, mask)
        
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)
        self.attention_scores = attention_scores
        
        return self.w_o(x)

    @staticmethod
    def _attention(query, key, value, mask, dropout=None):
        """Computes the scaled dot-product attention."""
        d_k = query.size(-1)
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, MultiheadAttention.MASK_VALUE)

        attention_scores = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return attention_scores @ value, attention_scores
