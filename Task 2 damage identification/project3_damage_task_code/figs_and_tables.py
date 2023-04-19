import numpy as np
import matplotlib.pyplot as plt
from fembeam import beam_fem
from numpy import linalg as LA
from oma import rand_vib
import scipy.io as io

### Table: Comparsion of predicted and measured first mode frequencies under damage case 2-11 ###
# A = np.array([1, 0.8, 0.96, 1.1, 0.48, 0.6, 0.4, 0.36, 0.53, 0.62])
# rho = np.array([1, 5.6, 4.2, 2.1, 3.0, 1.2, 3.4, 0.3, 1.5, 0.4])*1e4
# E = np.array([2, 3.4, 0.2, 0.8, 1.2, 4.2, 5.3, 6.6, 2.4, 0.5])*1e10
# I = np.array([8.33, 6.67, 8.00, 9.17, 4.00, 5, 3.33, 3.00, 4.33, 5.17])*1e-2
# alphas_set = np.array([[0.0, 0.2, 0.0],
#                        [0.0, 0.3, 0.0],
#                        [0.0, 0.4, 0.0],
#                        [0.1, 0.0, 0.0],
#                        [0.3, 0.0, 0.0],
#                        [0.5, 0.0, 0.0],
#                        [0.2, 0.2, 0.0],
#                        [0.2, 0.4, 0.0],
#                        [0.4, 0.2, 0.0],
#                        [0.4, 0.4, 0.0]
#                        ])
# beam = beam_fem(rho=rho[0], E=E[0], A=A[0], I=I[0], L=0.5)
# ratio = beam.freqs_ratio()
# pred_freq = []
# for i in range(10):
#     model_freq = beam.frequency(order=1, alphas=alphas_set[i, :])
#     pred_freq.append(round(model_freq*ratio, 4))
# print(pred_freq)


### Figure: Computed -log10(NRMSE) between each two different first mode shapes ###
# def nrmse(a, b):
#     return LA.norm(a-b, 2)/LA.norm(a, 2)
# A = np.array([1, 0.8, 0.96, 1.1, 0.48, 0.6, 0.4, 0.36, 0.53, 0.62])
# rho = np.array([1, 5.6, 4.2, 2.1, 3.0, 1.2, 3.4, 0.3, 1.5, 0.4])*1e4
# E = np.array([2, 3.4, 0.2, 0.8, 1.2, 4.2, 5.3, 6.6, 2.4, 0.5])*1e10
# I = np.array([8.33, 6.67, 8.00, 9.17, 4.00, 5, 3.33, 3.00, 4.33, 5.17])*1e-2
# phi = []
# for i in range(10):
#     beam = beam_fem(rho=rho[i], E=E[i], A=A[i], I=I[i], L=0.5)
#     ms_full = beam.modeshape(1, type='full')
#     phi.append(ms_full)
# phi = np.array(phi)
# nr = np.zeros((10, 10))
# for i in range(10):
#     for j in range(10):
#         nr[i, j] = nrmse(phi[i, :], phi[j, :])
# cm = 1 / 2.54
# fig = plt.figure(figsize=(10 * cm, 10 * cm))
# ax = fig.subplots()
# ax.matshow(nr, cmap='pink')
# plt.xticks(np.arange(10), np.arange(10)+1, fontname='Times New Roman')
# plt.yticks(np.arange(10), np.arange(10)+1, fontname='Times New Roman')
# for i in range(10):
#     for j in range(10):
#         if i == j:
#             pass
#         else:
#             c = '%.2f' % -np.log10(nr[j, i])
#             ax.text(i, j, str(c), va='center', ha='center',
#                     fontname='Times New Roman', fontsize=8)
# ax.xaxis.set_ticks_position('none')
# ax.yaxis.set_ticks_position('none')
# ax.tick_params(axis='x', labelsize=8)
# ax.tick_params(axis='y', labelsize=8)
# plt.savefig('./project3_damage_task_code/figs/F_nrmse_mtx.pdf', dpi=1200, bbox_inches='tight')
# plt.show()


### Figure: an instance of time-domain random vibration signals and the corresponding frequency domain representations ###
# mat = io.loadmat('./project3_damage_task_code/data/train_dataset/train_8.mat')
# mtx = mat['A']
# vib_analysis = rand_vib(signal_mtx=mtx)
# vib_analysis.plot_first_three_signal()
# plt.show()


### Figure: Bayesian Optimization convergence curve ###
# from skopt import load
# from skopt.plots import plot_convergence
# from neuralnets import Establish_nn_bayes_opt
# from matplotlib import pyplot as plt
# neural_net_path = r'./project3_damage_task_code/data/neural_nets/'
# res = load(neural_net_path+'hist.pkl')
# plot_convergence(res)
# plt.show()

### Figure: Neural networks training convergence curve ###
# from matplotlib import pyplot as plt
# import torch
# neural_net_path = r'./Task 2 damage identification/project3_damage_task_code/data/neural_nets/'
# train_loss = np.load(neural_net_path+'train_loss.npy')
# test_loss = np.load(neural_net_path+'test_loss.npy')
# cm = 1 / 2.54
# col = ['k', 'b', 'r']
# fig = plt.figure(figsize=(14 * cm, 10 * cm))
# ax = fig.subplots()
# ax.semilogy(train_loss,
#             color=col[1], dashes=[2, 2], label='train loss')
# ax.semilogy(test_loss,
#             color=col[2], label='test loss')
# legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0,
#                    handlelength=2.8, edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)
# legend.get_frame().set_boxstyle('Square', pad=0.0)
# legend.get_frame().set_lw(0.75)
# legend.get_frame().set_alpha(None)
# ax.set_xlabel('Epoch', fontname='Times New Roman', fontsize=8, labelpad=1)
# ax.set_ylabel('Loss', fontname='Times New Roman', fontsize=8, labelpad=1)
# ax.tick_params(axis='x', labelsize=8)
# ax.tick_params(axis='y', labelsize=8)
# ax.tick_params(axis='x', direction='in')
# ax.tick_params(axis='y', direction='in')
# ax.grid()
# ax.set_xlim([0, 200])
# ax.set_ylim([3e-6, 1e-2])
# plt.savefig('./Task 2 damage identification/project3_damage_task_code/figs/F_NN_converge.pdf', dpi=1200, bbox_inches='tight')
# plt.show()


### Table: Predicted damage factors from train_dataset 1-11 using neural networks###

# from oma import rand_vib
# from neuralnets import NeuralNetwork, Establish_nn_bayes_opt
# import os
# import torch
# from skopt import load
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# train_data_path = r'./project3_damage_task_code/data/train_dataset/'
# neural_net_path = r'./project3_damage_task_code/data/neural_nets/'
# res = load(neural_net_path+'hist.pkl')
# num_neuron, num_hidden_layer, learning_rate = res.x
# print(num_neuron, num_hidden_layer, learning_rate)
# # Neural network structure
# num_neuron = 381
# num_hidden_layer = 4
# nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
# net = NeuralNetwork(nn_structure)
# net.load_state_dict(torch.load(neural_net_path+'nn_model.pt'))
# # File names
# file_names = os.listdir(train_data_path)
# file_names.sort(key=lambda x: int(x[6:-4]))
# pred_list = []
# for i, file_name in enumerate(file_names):
#     file_path = train_data_path + file_name
#     mat = io.loadmat(file_path)
#     mtx = mat['A']
#     vib_analysis = rand_vib(signal_mtx=mtx)
#     ipt = vib_analysis.neur_net_input()
#     pred = net.forward(torch.tensor(ipt).float())
#     pred = torch.relu(pred)
#     print(pred.detach().numpy())
#     pred_list.append(pred.detach().numpy())
# pred_list = np.array(pred_list).reshape(11,3)
# ground_truth = np.genfromtxt(neural_net_path+'ground_truth.csv', delimiter=',')
# print(pred_list)
# print(np.sum(np.abs(pred_list-ground_truth), axis=None)/33)

