import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid


class LSTM(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_classes, n_layers=1, p_drop=.5):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.drop1 = nn.Dropout(p=p_drop)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.drop2 = nn.Dropout(p=p_drop)
        self.fc1 = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.drop3 = nn.Dropout(p=p_drop)
        self.fc2 = nn.Linear(
            in_features=hidden_dim,
            out_features=n_classes
        )

    def forward(self, x):
        x = x.transpose(-1, -2)
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x = torch.mean(x, dim=1).view(x.shape[0], -1)
        # x = x[:, -1].view(x.shape[0], -1)
        x = self.fc1(x)
        x = relu(self.bn(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x