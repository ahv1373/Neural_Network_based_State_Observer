import os

import torch

from src.python.neural_network_based_observer.MLP_architecture import MLPArchitecture

model = MLPArchitecture(input_size=4, hidden_size=64, output_size=4)
# load model with the path of os.path.join("output_models", "lstm_model_final.pt")
model.load_state_dict(torch.load(os.path.join("output_models", "lstm_model_final.pt")))
model.to('cpu')
x_input = torch.rand(1, 4)
output = model(x_input)
traced_model = torch.jit.trace(model.forward, x_input)
traced_model.save("output_models/lstm_model_final_traced.pt")