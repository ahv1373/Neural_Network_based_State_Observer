import torch
from torch import nn


# the purpose of this class is to define the MLP architecture for timeseries forecasting.
# The input of the model is the timeseries data of "s_x", "sx_int", and "omega".
# The output of the model is the timeseries data of "s_x", "sx_int", and "omega" at forecast_length seconds ahead.

class MLPArchitecture(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLPArchitecture, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out