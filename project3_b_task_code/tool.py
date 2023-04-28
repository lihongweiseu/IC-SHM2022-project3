# %% This code is for IC-SHM 2022 Data-driven modeling task

import time
import math
import numpy as np
import torch
from torch import nn, optim
from scipy.io import savemat, loadmat


# Prepare data
def data(root, input_type):
    global fi
    if input_type == 'clean':
        fi = np.transpose(loadmat(root + '/b/data_clean.mat')
                          ['data'])  # clean input data

    if input_type == 'noise':
        fi = np.transpose(loadmat(root + '/b/data_noised.mat')
                          ['data_noised'])  # noised input data

    fo = np.transpose(loadmat(root + '/b/data_clean.mat')
                      ['data'])  # output data

    u = np.hstack((fi[:, 0:1], fi[:, 1:2]))  # input for task B
    y_ref = np.hstack((fo[:, 2:3], fo[:, 3:4], fo[:, 4:5]))  # output for task B
    u_torch, y_ref_torch = torch.tensor(u), torch.tensor(y_ref)  # convert to tensor data

    return u, u_torch, y_ref, y_ref_torch


# Prepare models
def models(model_name, hidden_size, num_layers):
    global model
    input_size, output_size = 2, 3

    # Define BiLSTM
    class BiLstm(nn.Module):
        def __init__(self):
            super(BiLstm, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
            self.linear = nn.Linear(2*hidden_size, output_size)

        def forward(self, u):
            h0 = torch.zeros(2*num_layers, hidden_size)
            c0 = torch.zeros(2*num_layers, hidden_size)
            y, (hn, cn) = self.lstm(u, (h0, c0))
            y = self.linear(y)
            return y

    # Define LSTM
    class Lstm(nn.Module):
        def __init__(self):
            super(Lstm, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, u):
            h0 = torch.zeros(num_layers, hidden_size)
            c0 = torch.zeros(num_layers, hidden_size)
            y, (hn, cn) = self.lstm(u, (h0, c0))
            y = self.linear(y)
            return y

    # Define RNN
    class Rnn(nn.Module):
        def __init__(self):
            super(Rnn, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, u):
            h0 = torch.zeros(num_layers, hidden_size)
            y, hn = self.rnn(u, h0)
            y = self.linear(y)
            return y

    # Create model
    BiLSTM_model = BiLstm()
    LSTM_model = Lstm()
    RNN_model = Rnn()

    if model_name == 'biLSTM':
        model = BiLSTM_model
    if model_name == 'LSTM':
        model = LSTM_model
    if model_name == 'RNN':
        model = RNN_model

    return model


# define training function
def training(root, model_name, criterion, tend_train, num_layers, hidden_size, training_num, lr, device):

    # determine the model
    model = models(model_name, hidden_size, num_layers)

    # determine the training data
    u, u_torch, y_ref, y_ref_torch = data(root, 'clean')
    dt = 0.01  # 1/sampling frequency
    Nt_train = math.floor(tend_train / dt) + 1  # length of training data set
    u_train = torch.tensor(u[0:Nt_train, :])  # training input
    y_train_ref = torch.tensor(y_ref[0:Nt_train, :]).to(device)  # training output

    optimizer = optim.Adam(model.parameters(), lr)
    loss_all = np.zeros((training_num + 1, 1))
    start = time.time()

    for i in range(training_num):

        y_train_pred = model(u_train).to(device)
        loss = criterion(y_train_pred, y_train_ref)
        loss_all[i:i + 1, :] = loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0 or i == 0:
            print(f"iteration: {i + 1}, loss: {loss_all[i:i + 1, :].item()}")
            end = time.time()
            per_time = (end - start) / (i + 1)
            print("Average training time: %.6f s per one training" % per_time)
            print("Cumulative training time: %.6f s" % (end - start))
            left_time = (training_num - i + 1) * per_time
            print(f"Executed at {time.strftime('%H:%M:%S', time.localtime())},", "left time: %.6f s\n" % left_time)

    end = time.time()
    print("Total training time: %.3f s" % (end - start))

    # save model
    torch.save(model.state_dict(), root + "/model_checkpoint/" +
               str(model_name) + "_" + str(tend_train) + "s_" + str(training_num)+".pt")


# define validation function
def validation(root, model_name, criterion, tend_train, num_layers, hidden_size, training_num,
               input_type_valid, model_input):

    # determine the model
    global y_pred
    model = models(model_name, hidden_size, num_layers)

    # load model for validation
    model.load_state_dict(torch.load(root + "/model_checkpoint/" +
                                     str(model_name) + "_" + str(tend_train) + "s_" + str(training_num) + ".pt"))

    # load data for validation
    _, u_torch, _, y_ref_torch = data(root, input_type_valid)
    dt = 0.01  # 1/sampling frequency
    Nt_train = math.floor(tend_train / dt) + 1  # length of training data

    # prediction results
    # for supposed training length
    if model_input == 'training length':
        y_pred = model(u_torch[0:Nt_train, :])
        y_ref_torch = y_ref_torch[0:Nt_train, :]
    # for validate whole 10000-second length
    if model_input == 'whole length':
        y_pred = model(u_torch)

    # extract A3 to A5 prediction for task B
    y_pred_3, y_pred_4, y_pred_5 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    # extract A3 to A5 ground truth for task B
    y_ref_torch_3, y_ref_torch_4, y_ref_torch_5 = y_ref_torch[:, 0], y_ref_torch[:, 1], y_ref_torch[:, 2]
    # MSE for task B, e.g., the pair (y_ref_torch_3, y_pred_3) is for A3
    mse_3 = criterion(y_ref_torch_3, y_pred_3)
    mse_4 = criterion(y_ref_torch_4, y_pred_4)
    mse_5 = criterion(y_ref_torch_5, y_pred_5)
    print(f"MSE of Task 1b, A3: {mse_3.cpu().detach().numpy()}")
    print(f"MSE of Task 1b, A4: {mse_4.cpu().detach().numpy()}")
    print(f"MSE of Task 1b, A5: {mse_5.cpu().detach().numpy()}")


# define testing function
def testing(root, model_name, tend_train, num_layers, hidden_size, training_num):

    # determine the model
    model = models(model_name, hidden_size, num_layers)

    # load model for testing
    model.load_state_dict(torch.load(root + "/model_checkpoint/" +
                                     str(model_name) + "_" + str(tend_train) + "s_" + str(training_num) + ".pt"))
    # load data for testing
    fi = np.transpose(loadmat(root + "/b/data_noised_testset2.mat")
                      ['data_noised'])
    u = np.hstack((fi[:, 0:1], fi[:, 1:2]))  # input for task B
    u_torch = torch.tensor(u)  # convert to tensor data

    # prediction results
    y_pred = model(u_torch.float())

    return y_pred
