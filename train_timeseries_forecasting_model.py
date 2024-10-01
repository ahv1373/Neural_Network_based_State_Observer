import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from LSTM_architecture import LSTMArchitecture
from MLP_architecture import MLPArchitecture
from dataset_timeseries_loader import DatasetTimeseriesLoader

# Define Training Hyperparameters
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 64
SEQUENCE_LENGTH = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'output_models'

os.makedirs(SAVE_DIR, exist_ok=True)


# The purpose of this class is to train the LSTM model for timeseries forecasting

# read the dataset from the csv files
dataset_timeseries_loader = DatasetTimeseriesLoader(batch_size=BATCH_SIZE)
train_loader, test_loader = dataset_timeseries_loader.exec()

# define the LSTM model
MLP_model = MLPArchitecture(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(DEVICE)
MLP_model.train()

# define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = Adam(MLP_model.parameters(), lr=LEARNING_RATE)

# train the LSTM model
train_loss_list = []
test_loss_list = []
for epoch in tqdm(range(NUM_EPOCHS), desc='Training the model'):
    train_loss = 0.0
    for batch_idx, (input_tensor, output_tensor) in enumerate(train_loader):
        input_tensor, output_tensor = input_tensor.to(DEVICE), output_tensor.to(DEVICE)

        # Forward pass
        outputs = MLP_model(input_tensor)

        # Calculate the loss
        loss = criterion(outputs, output_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss_list.append(train_loss / len(train_loader))

    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input_tensor, output_tensor) in enumerate(test_loader):
            input_tensor, output_tensor = input_tensor.to(DEVICE), output_tensor.to(DEVICE)

            # Forward pass
            outputs = MLP_model(input_tensor)

            # Calculate the loss
            loss = criterion(outputs, output_tensor)

            test_loss += loss.item()

        test_loss_list.append(test_loss / len(test_loader))

    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss_list[-1]}, Test Loss: {test_loss_list[-1]}')

    # save the model every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(MLP_model.state_dict(), os.path.join(SAVE_DIR, f'lstm_model_{epoch + 1}.pt'))

# save the final model
torch.save(MLP_model.state_dict(), os.path.join(SAVE_DIR, 'lstm_model_final.pt'))
print('Finished Training')

# Plot the training and test loss
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

