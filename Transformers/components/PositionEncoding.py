import torch
import math

from torch.nn import Module
from torch import cos
from torch import sin

class PositionalEncoding(Module):
    def __init__(self, embedding_dim, max_token_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_token_len, embedding_dim)
        position = torch.arange(0, max_token_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * - (math.log(10000.0)/embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]