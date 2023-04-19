from oma import rand_vib
from neuralnets import NeuralNetwork
import os
import scipy.io as io
import torch
import numpy as np
# This line is added by Hong-Wei Li to test Git push
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

test_data_path = r'./Task 2 damage identification/project3_damage_task_code/data/test_dataset/'
neural_net_path = r'./Task 2 damage identification/project3_damage_task_code/data/neural_nets/'
num_neuron = 512
num_hidden_layer = 6
nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
net = NeuralNetwork(nn_structure)
net.load_state_dict(torch.load(
    neural_net_path+'nn_model_tanh512_6_1.1_1.0.pt', map_location=torch.device('cpu')))
file_names = os.listdir(test_data_path)
print(file_names)
pred_list = []
for i, file_name in enumerate(file_names):
    file_path = test_data_path + file_name
    mat = io.loadmat(file_path)
    mtx = mat['B']
    vib_analysis = rand_vib(signal_mtx=mtx)
    ipt = vib_analysis.neur_net_input()
    pred = net.forward(torch.tensor(ipt).float())
    pred = torch.relu(pred)
    print(pred.detach().numpy())
    pred_list.append(pred.detach().numpy())
pred_list = np.array(pred_list).reshape(6, 3)
np.savetxt(r'./'+'project3_damage_task.txt',
           pred_list, fmt='%f', delimiter=',')
