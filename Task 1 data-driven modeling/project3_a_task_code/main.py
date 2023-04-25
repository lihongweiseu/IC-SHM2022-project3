# %% This code is for IC-SHM 2022 Data-driven modeling task

import random
import numpy as np
import torch
from tool import training, validation

# Choose to use gpu or cpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# To guarantee same results for every running, which might slow down the training speed
torch.set_default_dtype(torch.float64)
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

# Define important configurations
task = 'A'  # define task is 'A'
input_type_train = 'clean'  # define whether clean or noised input is in training
input_type_valid = 'clean'  # define whether clean or noised input is in validation
tend_train = 10  # length of training data in seconds, must be integer times of dt and not greater than tend
criterion = torch.nn.MSELoss()  # define mean square error (MSE) loss
hidden_size = 40  # size of hidden state
num_layers = 1  # number of LSTM layers
lr = 0.01  # learning rate
training_num = 2000  # training iterations

# Determine the used model, model_name can be 'biLSTM', 'LSTM', or 'RNN'
model_name = 'biLSTM'

# Initialize training, if don't want to train pls annotate it
training(model_name, criterion, task, tend_train, input_type_train, device,
         num_layers, hidden_size, lr, training_num)

# Implement validation, calculate MSE, and prepare the prediction and ground truth data for drawing
# model_input is optional, including training length, whole length, or last 10s length
validation(model_name, criterion, task, tend_train, input_type_train, input_type_valid,
           num_layers, hidden_size, training_num, model_input='last 10s length')
