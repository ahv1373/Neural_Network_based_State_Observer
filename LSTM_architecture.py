import torch
from torch import nn


# the purpose of this class is to define the LSTM architecture for timeseries forecasting.
# The input of the model is the timeseries data of "s_x", "sx_int", and "omega".
# The output of the model is the timeseries data of "s_x", "sx_int", and "omega" at forecast_length seconds ahead.
class LSTMArchitecture(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMArchitecture, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out
    