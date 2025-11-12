import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)/d_model)))
        
        #Needs to be modified to support the lattice QCD data
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(poisition * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  #makes it an unlearnable parameter 
    
    def forward(self, x):
        return x + self.pe[:, x.size(1)]
