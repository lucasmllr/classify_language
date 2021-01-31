import torch.nn as nn
from torch.nn.functional import relu


class CharConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_kernel=None, norm=False):
        super(CharConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding=padding    
        )
        if norm:
            self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.relu =  nn.ReLU()
        if pool_kernel is not None:
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel)
        
    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = self.relu(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        return x


class CharFC(nn.Module):
    def __init__(self, in_features, out_features, p_dropout=None):
        super(CharFC, self).__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features
        )
        self.relu = nn.ReLU()
        if p_dropout is not None:
            self.drop = nn.Dropout(p=p_dropout)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        if hasattr(self, 'drop'):
            x = self.drop(x)
        return x


class CharCNN(nn.Module):
    """character level CNN based on https://arxiv.org/pdf/1509.01626.pdf
    """
    def __init__(self, vocab_len, conv_features, fc_in_features, fc_features, n_classes):
        super(CharCNN, self).__init__()
        self.conv1 = CharConv(vocab_len, conv_features, kernel_size=7, padding=3, pool_kernel=3)
        self.conv2 = CharConv(conv_features, conv_features, kernel_size=7, padding=3, pool_kernel=3)
        self.conv3 = CharConv(conv_features, conv_features, kernel_size=3, padding=1)
        self.conv4 = CharConv(conv_features, conv_features, kernel_size=3, padding=1)
        self.conv5 = CharConv(conv_features, conv_features, kernel_size=3, padding=1)
        self.conv6 = CharConv(conv_features, conv_features, kernel_size=3, padding=1, pool_kernel=3)
        self.fc1 = CharFC(fc_in_features, fc_features, p_dropout=.5)
        self.fc2 = CharFC(fc_features, fc_features, p_dropout=.5)
        self.fc3 = CharFC(fc_features, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CharnkocCNN(nn.Module):

    def __init__(self, len_vocab=25, n_classes=14, input_length=1014, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))

        dimension = int((input_length - 96) / 27 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

    