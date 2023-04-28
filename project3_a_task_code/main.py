# %% This code is for IC-SHM 2022 Data-driven modeling task

import random
import numpy as np
import torch
from tool import training, validation

# Choose to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# To guarantee same results for every running, which might slow down the training speed
torch.set_default_dtype(torch.float64)
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

# Define the root explorer, please change to the root in your conditions
root = '.'

# Define important configurations
criterion = torch.nn.MSELoss()  # Define mean square error (MSE) loss.
tend_train = 10  # Length of training data in seconds.
hidden_size = 40  # Size of hidden state.
num_layers = 1  # Number of LSTM layers.
lr = 0.01  # Learning rate.
training_num = 2000  # Training iterations.

# Determine the used model, model_name can be 'biLSTM', 'LSTM', or 'RNN'.
model_name = 'biLSTM'

# Initialize training, if don't want to train please annotate it.
# training(root, model_name, criterion, tend_train, num_layers, hidden_size, training_num, lr, device)

# Input for validation is optional, it can be 'clean' or 'noise'.
input_type_valid = 'noise'

# Model_input is optional, it can be 'training length' or 'whole length'.
model_input = 'whole length'

# Implement validation, calculate MSE, and prepare the prediction and ground truth data for drawing.
validation(root, model_name, criterion, tend_train, num_layers, hidden_size, training_num, input_type_valid, model_input)
