# %% This code is for IC-SHM 2022 Data-driven modeling task
# testing for generating task A predictions

import numpy as np
from tool import testing
from torch import nn

task = 'A'  # task is 'A'
input_type_train = 'clean'  # use model by training input from clean or noise
criterion = nn.MSELoss()  # MSE loss
hidden_size = 40
num_layers = 1
training_num = 2000
tend_train = 10
model_name = 'biLSTM'

y_pred = testing(model_name, task, tend_train, input_type_train, num_layers, hidden_size, training_num)
y = np.transpose(y_pred.detach().numpy())

save_path = './project3_a_task.txt'
np.savetxt(save_path, y, fmt='%f', delimiter=",")
