import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, learnable=False):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0))/d_model))
        
        #Sinusoidal positional encoding but may be made learnable to help model performance
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if learnable: # may be more computationally expensive but it can improve performance
            self.pe = nn.Parameter(pe.unsqueeze(0))
        else:
            self.register_buffer('pe', pe.unsqueeze(0))  #makes it an unlearnable parameter

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
