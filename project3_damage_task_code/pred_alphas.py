# This file is used to predict the alpha values for 
# the train_dataset using the neural network model.
from oma import rand_vib
from neuralnets import NeuralNetwork, Establish_nn_bayes_opt
import os
import scipy.io as io
import torch
import numpy as np
# from skopt import load

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_data_path = r'./damage identification task/data/train_dataset/'
neural_net_path = r'./damage identification task/data/neural_nets/'
# res = load(neural_net_path+'hist.pkl')
# num_neuron, num_hidden_layer, learning_rate = res.x
# print(num_neuron, num_hidden_layer, learning_rate)
num_neuron = 381
num_hidden_layer = 4
nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
net = NeuralNetwork(nn_structure)
net.load_state_dict(torch.load(neural_net_path+'nn_model.pt'))

file_names = os.listdir(train_data_path)
file_names.sort(key=lambda x: int(x[6:-4]))
pred_list = []
for i, file_name in enumerate(file_names):
    file_path = train_data_path + file_name
    mat = io.loadmat(file_path)
    mtx = mat['A']
    vib_analysis = rand_vib(signal_mtx=mtx)
    ipt = vib_analysis.neur_net_input()
    pred = net.forward(torch.tensor(ipt).float())
    pred = torch.relu(pred)
    print(pred.detach().numpy())
    pred_list.append(pred.detach().numpy())

pred_list = np.array(pred_list).reshape(11,3)
ground_truth = np.genfromtxt(neural_net_path+'ground_truth.csv', delimiter=',')
# evaluate the performance
print(pred_list)
print(np.sum(np.abs(pred_list-ground_truth), axis=None)/33)
