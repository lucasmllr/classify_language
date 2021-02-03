import torch
import torch.nn as nn


class BOWClassifier(nn.Module):

    def __init__(self, vocab_len, hidden_dim, n_classes):
        super(BOWClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_len + 1, hidden_dim, padding_idx=vocab_len)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.emb(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)