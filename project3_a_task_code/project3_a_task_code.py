# %% This code is for IC-SHM 2022 Data-driven modeling task
# testing for generating task A predictions

import numpy as np
from tool import testing
from torch import nn

root = '.'  # You must set the folder IC-SHM2022-project3 as the workspace.
criterion = nn.MSELoss()  # MSE loss
tend_train = 10
hidden_size = 40
num_layers = 1
training_num = 2000
model_name = 'biLSTM'

y_pred = testing(root, model_name, tend_train,
                 num_layers, hidden_size, training_num)
y = np.transpose(y_pred.detach().numpy())

save_path = root + '/project3_a_task_code/project3_a_task.txt'
np.savetxt(save_path, y, fmt='%f', delimiter=",")
