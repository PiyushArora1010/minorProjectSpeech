import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)


class DeepLearningModel(nn.Module):
    def __init__(self, config):
        super(DeepLearningModel, self).__init__()
        self.config = self._readConfig(config)
        self.ConvBlocks = nn.ModuleList()
        self.LSTMBlocks = nn.ModuleList()
        self.LinearBlocks = nn.ModuleList()

        for params in self.config['Conv1d']:
            self.ConvBlocks.append(ConvBlock(*params))
        for params in self.config['LSTM']:
            self.LSTMBlocks.append(nn.LSTM(*params))
            self.LSTMBlocks[-1].batch_first = True
        for params in self.config['Linear']:
            self.LinearBlocks.append(nn.Linear(*params))

        self.ConvBlocks = nn.Sequential(*self.ConvBlocks)
        self.LSTMBlocks = nn.Sequential(*self.LSTMBlocks)
        self.LinearBlocks = nn.Sequential(*self.LinearBlocks)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.ConvBlocks(x)
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.LSTMBlocks(x)
        x = x[:, -1, :]
        x = self.LinearBlocks(x)
        return x

    def _readConfig(self,config):
        file = open(config, 'r')
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        config = {}
        config['Conv1d'] = []
        config['LSTM'] = []
        config['Linear'] = []
        for line in lines:
            line = line.split(':')
            if line[0] == 'Conv1d':
                config['Conv1d'].append([int(i) for i in line[1].split(',')])
            elif line[0] == 'LSTM':
                config['LSTM'].append([int(i) for i in line[1].split(',')])
            elif line[0] == 'Linear':
                config['Linear'].append([int(i) for i in line[1].split(',')])
        return config
    