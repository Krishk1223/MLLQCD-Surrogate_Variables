import torch.nn as nn 

class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2) #changes shape to be by (batch_size x d_model x seq_len) for pooling
        x = self.pool(x) #pools over seq_len dimension to get (batch_size x d_model x 1)
        x = x.squeeze(-1) #last dimension removal (batch_size x d_model)
        # x = x.mean(dim=1) #alternative pooling method/ probs faster
        return self.fc(x)
        