import glob

import torch
from torch import nn
import os

from torch.utils.data import DataLoader, TensorDataset

from dataset_timeseries_loader import DatasetTimeseriesLoader
import matplotlib.pyplot as plt

from src.python.neural_network_based_observer.MLP_architecture import MLPArchitecture

# the purpose of this class is to define the MLP architecture for timeseries forecasting. Load the trained model and use it for time series forecasting.



# Load the data
dataset_timeseries_loader = DatasetTimeseriesLoader()
target_csv_path = os.path.join("dataset", "unseen_testing_set", "SL_recording_passive_2_a_x.csv")
timeseries = dataset_timeseries_loader.load_timeseries(target_csv_path)
timeseries = timeseries[['a_x', 's_x', 's_x_int', 'omega']]
input_timeseries, output_timeseries = dataset_timeseries_loader.get_input_output_timeseries(timeseries)
test_loader = DataLoader(TensorDataset(input_timeseries, output_timeseries), batch_size=1, shuffle=False)

model = MLPArchitecture(input_size=4, hidden_size=64, output_size=4)
# load model with the path of os.path.join("output_models", "lstm_model_final.pt")
model.load_state_dict(torch.load(os.path.join("output_models", "lstm_model_final.pt")))
model.eval()
model.to(dataset_timeseries_loader.device)

# Forecast the timeseries and plot the prediction of each signal vs the ground truth
a_x_dict = {'ground_truth': [], 'prediction': []}
s_x_dict = {'ground_truth': [], 'prediction': []}
sx_int_dict = {'ground_truth': [], 'prediction': []}
omega_dict = {'ground_truth': [], 'prediction': []}
with torch.no_grad():
    for i, (input_tensor, output_tensor) in enumerate(test_loader):
        input_tensor, output_tensor = input_tensor.to(dataset_timeseries_loader.device), output_tensor.to(
            dataset_timeseries_loader.device)

        # Forward pass
        outputs = model(input_tensor)

        a_x_dict['ground_truth'].extend(output_tensor[:, 0].cpu().numpy())
        a_x_dict['prediction'].extend(outputs[:, 0].cpu().numpy())
        s_x_dict['ground_truth'].extend(output_tensor[:, 1].cpu().numpy())
        s_x_dict['prediction'].extend(outputs[:, 1].cpu().numpy())
        sx_int_dict['ground_truth'].extend(output_tensor[:, 2].cpu().numpy())
        sx_int_dict['prediction'].extend(outputs[:, 2].cpu().numpy())
        omega_dict['ground_truth'].extend(output_tensor[:, 3].cpu().numpy())
        omega_dict['prediction'].extend(outputs[:, 3].cpu().numpy())

# [['a_x', 's_x', 's_x_int', 'omega']]
# Open a plt subplot of 2x1 and plot the prediction of each signal vs the ground truth
fig, axs = plt.subplots(4, 2, figsize=(10, 15))
time_data = [i * dataset_timeseries_loader.time_step for i in range(len(s_x_dict['ground_truth']))]

# in axs[0, 0], plot the first input signal to the model
axs[0, 0].plot(time_data, input_timeseries[:, 0].cpu().numpy(), color='green', linestyle='dashed', linewidth=2)
axs[0, 0].set_title('Current a_x')
axs[0, 0].grid()


axs[0, 1].plot(time_data, a_x_dict['ground_truth'], label='Ground Truth', color='blue', linestyle='dashed', linewidth=4)
axs[0, 1].plot(time_data, a_x_dict['prediction'], label='Prediction', color='red', linewidth=2)
axs[0, 1].set_title('a_x')
axs[0, 1].legend()
axs[0, 1].grid()

axs[1, 0].plot(time_data, input_timeseries[:, 1].cpu().numpy(), color='green', linestyle='dashed', linewidth=2)
axs[1, 0].set_title('Current s_x')
axs[1, 0].grid()

axs[1, 1].plot(time_data, s_x_dict['ground_truth'], label='Ground Truth', color='blue', linestyle='dashed', linewidth=4)
axs[1, 1].plot(time_data, s_x_dict['prediction'], label='Prediction', color='red', linewidth=2)
axs[1, 1].set_title('s_x')
axs[1, 1].legend()
axs[1, 1].grid()

axs[2, 0].plot(time_data, input_timeseries[:, 2].cpu().numpy(), color='green', linestyle='dashed', linewidth=2)
axs[2, 0].set_title('Current s_x_int')
axs[2, 0].grid()

axs[2, 1].plot(time_data, sx_int_dict['ground_truth'], label='Ground Truth', color='blue', linestyle='dashed', linewidth=4)
axs[2, 1].plot(time_data, sx_int_dict['prediction'], label='Prediction', color='red', linewidth=2)
axs[2, 1].set_title('s_x_int')
axs[2, 1].legend()
axs[2, 1].grid()

axs[3, 0].plot(time_data, input_timeseries[:, 3].cpu().numpy(), color='green', linestyle='dashed', linewidth=2)
axs[3, 0].set_title('Current omega')
axs[3, 0].grid()

axs[3, 1].plot(time_data, omega_dict['ground_truth'], label='Ground Truth', color='blue', linestyle='dashed', linewidth=4)
axs[3, 1].plot(time_data, omega_dict['prediction'], label='Prediction', color='red', linewidth=2)
axs[3, 1].set_title('omega')
axs[3, 1].legend()
axs[3, 1].grid()

plt.tight_layout()
plt.show()



