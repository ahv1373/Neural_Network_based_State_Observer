from typing import Tuple

import pandas as pd
import torch
import os
import glob

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# The purpose of this class is to load the dataset timeseries from the csv files
# Each csv file contains the columns of "Time", "s_x", "sx_int", and "omega".
# The outputs of the model are "s_x", "sx_int", and "omega" at forecast_length seconds ahead.
# Therefore, data shuffle for training and testing is not necessary as it is a timeseries data.

class DatasetTimeseriesLoader:
    def __init__(self, data_directory: str = os.path.join(os.path.dirname(__file__), 'dataset'),
                 forecast_length: float = 0.4, time_step: float = 0.001, **kwargs):
        self.data_directory = data_directory
        self.all_csv_timeseries_files = glob.glob(os.path.join(data_directory, '*.csv'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # forecast_length is the number of seconds ahead to forecast the timeseries
        self.forecast_length = forecast_length
        # time_step is the time step of the timeseries data
        self.time_step = time_step
        self.test_ratio = kwargs.get('test_ratio', 0.2)
        self.batch_size = kwargs.get('batch_size', 64)
        self.sequence_length = int(forecast_length / time_step)

    def load_timeseries(self, file_name: str) -> pd.DataFrame:
        return pd.read_csv(file_name)

    def get_input_output_timeseries(self, timeseries: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        # The input timeseries is the timeseries data from the beginning to the end - forecast_length
        input_timeseries = timeseries.iloc[:-self.sequence_length]
        # The output timeseries is the timeseries data from forecast_length to the end
        output_timeseries = timeseries.iloc[self.sequence_length:]
        # Convert the input and output timeseries to torch.Tensor
        input_timeseries = torch.tensor(input_timeseries.values, dtype=torch.float32, device=self.device)
        output_timeseries = torch.tensor(output_timeseries.values, dtype=torch.float32, device=self.device)
        return input_timeseries, output_timeseries

    def create_torch_data_loader(self, train_input_tensor, train_output_tensor, test_input_tensor,
                                 test_output_tensor) -> Tuple[DataLoader, DataLoader]:
        # Create the data loader
        train_dataset = TensorDataset(train_input_tensor, train_output_tensor)
        test_dataset = TensorDataset(test_input_tensor, test_output_tensor)
        # Dataloader should consider batch size, sequence length, and shuffle
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return train_data_loader, test_data_loader

    def exec(self) -> Tuple[DataLoader, DataLoader]:
        input_tensor, output_tensor = torch.tensor([]), torch.tensor([])
        for csv_file in self.all_csv_timeseries_files:
            timeseries = self.load_timeseries(csv_file)
            # Only select columns of "Time", "s_x", "sx_int", and "omega"
            timeseries = timeseries[['a_x', 's_x', 's_x_int', 'omega']]
            input_timeseries, output_timeseries = self.get_input_output_timeseries(timeseries)
            input_tensor = torch.cat((input_tensor, input_timeseries), dim=0)
            output_tensor = torch.cat((output_tensor, output_timeseries), dim=0)
        train_timeseries, test_timeseries = train_test_split(pd.concat([pd.DataFrame(input_tensor),
                                                                        pd.DataFrame(output_tensor)], axis=1),
                                                             test_size=self.test_ratio)
        train_input_tensor, train_output_tensor = torch.tensor(train_timeseries.iloc[:, :4].values, dtype=torch.float32,
                                                               device=self.device), \
            torch.tensor(train_timeseries.iloc[:, 4:].values, dtype=torch.float32,
                         device=self.device)
        test_input_tensor, test_output_tensor = torch.tensor(test_timeseries.iloc[:, :4].values, dtype=torch.float32,
                                                             device=self.device), \
            torch.tensor(test_timeseries.iloc[:, 4:].values, dtype=torch.float32,
                         device=self.device)
        return self.create_torch_data_loader(train_input_tensor, train_output_tensor, test_input_tensor,
                                             test_output_tensor)


if __name__ == '__main__':
    dataset_timeseries_loader = DatasetTimeseriesLoader()
    train_data_loader, test_data_loader = dataset_timeseries_loader.exec()
