import torch.nn as nn
from .MultiHeadAttention import MultiHeadAttention
from .FeedForwardSublayer import FeedForwardSublayer
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff_sublayer = FeedForwardSublayer(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        #Pre-Attention Layer Normalisation
        normalised_x = self.norm1(x)
        attn_out = self.self_attention(normalised_x, normalised_x, normalised_x, src_mask)
        x = x + self.dropout(attn_out)

        normalised_x = self.norm2(x)
        ff_out = self.ff_sublayer(normalised_x)
        x = x + self.dropout(ff_out)
        return x
