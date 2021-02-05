import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_classes, n_layers=1, p_drop=.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.drop = nn.Dropout(p=p_drop)
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=n_classes
        )

    def forward(self, x):
        x = x.transpose(-1, -2)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1).view(x.shape[0], -1)
        # x = x[:, -1].view(x.shape[0], -1)
        x = self.drop(x)
        x = self.fc(x)
        return x