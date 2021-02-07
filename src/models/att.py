import torch
from torch.nn.functional import relu
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, seq_len, embed_dim, n_heads, hidden_dim, n_classes, p_drop=0):
        super(Attention, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, n_heads, p_drop)
        self.fc1 = nn.Linear(seq_len * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.drop1 = nn.Dropout(p_drop)
        self.drop2 = nn.Dropout(p_drop)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # shape (L, N, embed_dim) for attention layer
        x = x.permute(2, 0, 1)
        ao, _ = self.att(x, x, x)
        ao = self.drop1(ao)
        x = self.norm1(ao + x)
        # shape (N, -1) for fc layers
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = relu(self.norm2(x))
        x = self.fc2(x)
        return x