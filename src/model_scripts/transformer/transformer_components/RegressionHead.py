import torch.nn as nn 

class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        # Sequence-to-sequence: project each time step from d_model to output_dim
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Input: (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, output_dim)
        return self.fc(x)
        