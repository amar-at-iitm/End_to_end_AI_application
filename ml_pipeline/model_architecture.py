# ml_pipeline/model_architecture.py
import torch
import torch.nn as nn

class StockPriceModel(nn.Module):
    def __init__(self, input_size):
        super(StockPriceModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # take the last timestep output
        out = self.fc(x)
        return out
