from skopt import load
import torch
import os
from neuralnets import NeuralNetwork, Establish_nn_bayes_opt
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
# alphas_set = np.array([[0.0, 0.0, 0.0],
#                        [0.0, 0.2, 0.0],
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
# freq = []
# for i in range(11):
#     model_freq = beam.frequency(order=1, alphas=alphas_set[i, :])
#     freq.append(round(model_freq, 4))
# freq_ans = np.array([9.4367, 9.3613, 9.3093, 9.2421, 9.4251,
#                     9.3924, 9.3345, 9.3356, 9.2167, 9.2933, 9.1751])
# print(np.array(freq)/freq_ans)
# print(freq)


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
# plt.savefig('./Task 2 damage identification/project3_damage_task_code/figs/F_nrmse_mtx.pdf',
#             dpi=1200, bbox_inches='tight')
# plt.show()


### Figure: an instance of time-domain random vibration signals and the corresponding frequency domain representations ###
# mat = io.loadmat(
#     r'./Task 2 damage identification/project3_damage_task_code/data/train_dataset/train_8.mat')
# mtx = mat['A']
# vib_analysis = rand_vib(signal_mtx=mtx)
# num = 2000
# T = np.linspace(0, (num-1)/100, num)
# col = ['k', 'b', 'r']
# cm = 1 / 2.54
# fig = plt.figure(figsize=(16 * cm, 14 * cm))
# ax = fig.add_subplot(211)
# ax.plot(T, vib_analysis.signal_mtx[0, 0:num],
#         color=col[0], label='First sensor')
# ax.plot(T, vib_analysis.signal_mtx[1, 0:num], color=col[1], dashes=[
#         8, 4], label='Second sensor')
# ax.plot(T, vib_analysis.signal_mtx[2, 0:num], color=col[2], dashes=[
#         2, 2], label='Third sensor', zorder=1)
# for line in ax.get_lines():
#     line.set_linewidth(0.5)
# ax.set_xlabel(r'Time (s)', fontsize=8, labelpad=1)
# ax.set_ylabel(
#     r'Acceleration ($\mathregular{m/s^2}$)', fontsize=8, labelpad=1)
# ax.set_ylim([-0.3, 0.3])
# ax.set_yticks(np.arange(-0.3, 0.31, 0.1))
# ax.set_xlim([0, 20])
# ax.set_xticks(np.arange(0, 20.1, 5))
# ax.tick_params(axis='x', labelsize=8)
# ax.tick_params(axis='y', labelsize=8)
# legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0,
#                    handlelength=2.8, edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)
# legend.get_frame().set_boxstyle('Square', pad=0.0)
# legend.get_frame().set_lw(0.75)
# legend.get_frame().set_alpha(None)
# for obj in legend.legendHandles:
#     obj.set_lw(0.75)
# ax.text(-1.2, -0.37, '(a)', fontsize=8)
# ax.tick_params(axis='x', direction='in')
# ax.tick_params(axis='y', direction='in')
# ax.grid()

# ax = fig.add_subplot(212)
# f1, pxx1 = vib_analysis.psd_analysis(dim=0)
# f2, pxx2 = vib_analysis.psd_analysis(dim=1)
# f3, pxx3 = vib_analysis.psd_analysis(dim=2)
# ax.semilogy(f1, pxx1, color=col[0], lw=1, label='First sensor')
# ax.semilogy(f2, pxx2, color=col[1], dashes=[
#     8, 4], lw=1, label='Second sensor')
# ax.semilogy(f3, pxx3, color=col[2], dashes=[
#     2, 2], lw=1, label='Third sensor')
# ax.set_xlabel(r'Frequency (Hz)', fontsize=8, labelpad=1)
# ax.set_ylabel(
#     r'PSD ($\mathregular{(m/s^2)^2}$/Hz)', fontsize=8, labelpad=1)
# ax.set_xlim([5, 40])
# ax.set_xticks(np.arange(5, 40.1, 5))
# ax.set_ylim([1e-7, 1e-1])
# # ax.set_yticks(np.arange(-7, -0.9, 1))
# ax.tick_params(axis='x', labelsize=8)
# ax.tick_params(axis='y', labelsize=8)
# legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0, handlelength=2.8,
#                    edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)  # labelspacing=0
# legend.get_frame().set_boxstyle('Square', pad=0.0)
# legend.get_frame().set_lw(0.75)
# legend.get_frame().set_alpha(None)
# for obj in legend.legendHandles:
#     obj.set_lw(0.75)
# ax.text(3, 10**(-7.6), '(b)', fontsize=8)
# ax.tick_params(axis='x', direction='in')
# ax.tick_params(axis='y', direction='in')
# ax.tick_params(axis='y', which='minor', direction='in')
# ax.grid()
# fig.tight_layout(pad=0.1)
# plt.savefig(r'./Task 2 damage identification/project3_damage_task_code/figs/F_threesignal.pdf',  format="pdf",
#             dpi=1200)
# plt.show()


### Figure: Bayesian Optimization convergence curve ###
# from skopt import load
# from skopt.plots import plot_convergence
# from neuralnets import Establish_nn_bayes_opt
# from matplotlib import pyplot as plt
# neural_net_path = r'./Task 2 damage identification/project3_damage_task_code/data/neural_nets/'
# res_relu = load(neural_net_path+'hist_relu.pkl')
# res_sigmoid = load(neural_net_path+'hist_sigmoid.pkl')
# res_tanh = load(neural_net_path+'hist_tanh.pkl')
# print(dir(res_relu))
# print(res_tanh.x)
# # print(vars(res_relu))
# # print(res_relu.func_vals)
# relu_vec = []
# sigmoid_vec = []
# tanh_vec = []
# for i in range(15):
#     relu_vec.append(np.min(res_relu.func_vals[:i+1]))
#     sigmoid_vec.append(np.min(res_sigmoid.func_vals[:i+1]))
#     tanh_vec.append(np.min(res_tanh.func_vals[:i+1]))

# col = ['k', 'b', 'r']
# cm = 1 / 2.54
# fig = plt.figure(figsize=(10 * cm, 7 * cm))
# ax = fig.subplots()
# x = np.arange(15)+1
# ax.semilogy(x, 10**np.array(relu_vec), color=col[0], label='ReLU', marker='o')
# ax.semilogy(x, 10**np.array(sigmoid_vec),
#             color=col[1], label='Sigmoid', marker='^')
# ax.semilogy(x, 10**np.array(tanh_vec), color=col[2], label='Tanh', marker='s')

# ax.set_xlabel(r'Number of calls n',
#               fontname='Times New Roman', fontsize=8, labelpad=1)
# ax.set_ylabel(r'Minimum of MSE after n calls',
#               fontname='Times New Roman', fontsize=8, labelpad=1)
# ax.set_ylim([1e-7, 5e-1])
# # ax.set_yticks(np.arange(-0.3, 0.31, 0.1))
# ax.set_xlim([0, 16])
# ax.set_xticks(np.arange(0, 16.1, 3))
# ax.tick_params(axis='x', labelsize=8, direction='in')
# ax.tick_params(axis='y', labelsize=8, direction='in')
# legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0, handlelength=2.8,
#                    edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)  # labelspacing=0
# legend.get_frame().set_boxstyle('Square', pad=0.0)
# legend.get_frame().set_lw(0.75)
# legend.get_frame().set_alpha(None)
# for obj in legend.legendHandles:
#     obj.set_lw(0.75)
# ax.grid()
# ax.tick_params(axis='y', which='minor', direction='in')
# fig.tight_layout(pad=0.1)
# plt.savefig('./Task 2 damage identification/project3_damage_task_code/figs/F_bayesopt.pdf',
#             dpi=1200)
# plt.show()

### Figure: Neural networks training convergence curve ###
# from matplotlib import pyplot as plt
# import torch
# neural_net_path = r'./Task 2 damage identification/project3_damage_task_code/data/neural_nets/'
# train_loss = np.load(neural_net_path+'train_loss_tanh512_6_1.1_1.0.npy')
# test_loss = np.load(neural_net_path+'test_loss_tanh512_6_1.1_1.0.npy')
# x = np.arange(40)+1
# cm = 1 / 2.54
# col = ['k', 'b', 'r']
# fig = plt.figure(figsize=(10 * cm, 10 * cm))
# ax = fig.subplots()
# ax.semilogy(x, train_loss,
#             color=col[1], dashes=[2, 2], label='Train loss')
# ax.semilogy(x, test_loss,
#             color=col[2], label='Test loss')
# legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0,
#                    handlelength=2.8, edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)
# legend.get_frame().set_boxstyle('Square', pad=0.0)
# legend.get_frame().set_lw(0.75)
# legend.get_frame().set_alpha(None)
# ax.set_xlabel('Epoch', fontname='Times New Roman', fontsize=8, labelpad=1)
# ax.set_ylabel('MSE', fontname='Times New Roman', fontsize=8, labelpad=1)
# ax.tick_params(axis='x', labelsize=8)
# ax.tick_params(axis='y', labelsize=8)
# ax.tick_params(axis='x', direction='in')
# ax.tick_params(axis='y', direction='in')
# ax.grid()
# ax.set_xlim([0, 40])
# ax.set_ylim([1e-7, 1e-3])
# plt.savefig('./Task 2 damage identification/project3_damage_task_code/figs/F_NN_converge.pdf',
#             dpi=1200, bbox_inches='tight')
# plt.show()


### Table: Predicted damage factors from train_dataset 1-11 using neural networks###

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# train_data_path = r'./Task 2 damage identification/project3_damage_task_code/data/train_dataset/'
# neural_net_path = r'./Task 2 damage identification/project3_damage_task_code/data/neural_nets/'
# # Neural network structure
# num_neuron = 512
# num_hidden_layer = 6
# nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
# net = NeuralNetwork(nn_structure, act_fun='tanh')
# net.load_state_dict(torch.load(
#     neural_net_path+'nn_model_tanh512_6_1.1_1.0.pt', map_location=torch.device('cpu')))
# # net = NeuralNetwork(nn_structure, act_fun='relu')
# # net.load_state_dict(torch.load(
# #     neural_net_path+'nn_model_relu512_6_1.0_1.0.pt', map_location=torch.device('cpu')))
# # File names
# file_names = os.listdir(train_data_path)
# file_names.sort(key=lambda x: int(x[6:-4]))
# pred_list = []
# for i, file_name in enumerate(file_names):
#     file_path = train_data_path + file_name
#     mat = io.loadmat(file_path)
#     mtx = mat['A']
#     vib_analysis = rand_vib(signal_mtx=mtx[:, 0:50_000])
#     ipt = vib_analysis.neur_net_input()
#     pred = net.forward(torch.tensor(ipt).float())
#     pred = torch.relu(pred)
#     # print(pred.detach().numpy())
#     pred_list.append(pred.detach().numpy())
# pred_list = np.array(pred_list).reshape(11, 3)
# ground_truth = np.genfromtxt(neural_net_path+'ground_truth.csv', delimiter=',')
# # print(pred_list)
# print((pred_list-ground_truth)*100)
# print(np.sum(np.abs(pred_list-ground_truth), axis=None)/33)
