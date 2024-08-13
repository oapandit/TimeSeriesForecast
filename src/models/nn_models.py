import torch.nn as nn
import torch

class FeedForwardModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(FeedForwardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(CNNModel, self).__init__()
        out_channels = 64
        self.conv1d = nn.Conv1d(input_size, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(out_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv1d(x.squeeze(1).permute(1,0))
        x = self.relu(x)
        x = x.permute(1,0)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TransformerDecoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead):
        super(TransformerDecoderModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        memory = torch.zeros_like(x)
        x = self.transformer_decoder(x, memory)
        x = self.fc(x[:, -1, :])
        return x

