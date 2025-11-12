import torch.nn as nn
import torch.functional as F
import math


class InputEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #standard scaling practice
