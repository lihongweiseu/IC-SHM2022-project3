import numpy as np
import torch
import torch.nn as nn
from fembeam import beam_fem
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Generate training set
# grid_num = 101
# alphas = np.linspace(0.1, 1, grid_num)
# alphas = np.array(np.meshgrid(alphas, alphas, alphas)
#                   ).reshape(3, -1).transpose()
# alphas = np.ones_like(alphas)-alphas
# beam = beam_fem()
# md_1st_r = []
# ms_ratio_undam = beam.md1st_ratio()
# for i in range(grid_num**3):
#     md_1st_r.append(beam.nn_input(alphas[i, :], ms_ratio_undam))
#     if i % (grid_num**2) == 0:
#         print("%.2f" % ((i/(grid_num**3))*100), '%')
# md_1st_r = np.array(md_1st_r)
# train_data = TensorDataset(torch.tensor(md_1st_r), torch.tensor(alphas))
# torch.save(
#     train_data, './damage identification task/data/neural_nets/train_data.pt')

# Generate test set
# alphas_test = np.random.rand(300000, 3)*0.9
# md_1st_r_test = []
# for i in range(len(alphas_test)):
#     md_1st_r_test.append(beam.nn_input(alphas_test[i, :], ms_ratio_undam))
#     if i % (3000) == 0:
#         print("%.2f" % (i/(3000)), '%')
# md_1st_r_test = np.array(md_1st_r_test)
# test_data = TensorDataset(torch.tensor(md_1st_r_test),
#                           torch.tensor(alphas_test))
# torch.save(test_data, './damage identification task/data/neural_nets/test_data.pt')

# neural network


class NeuralNetwork(nn.Module):
    def __init__(self, in_hid_out):
        # in_hid_out: a list of integers, the number of nodes for each layer
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.ModuleList()
        for i in range(len(in_hid_out)-2):
            self.mlp.append(
                nn.Linear(in_hid_out[i], in_hid_out[i+1], bias=False))
            self.mlp.append(nn.Tanh())
        self.mlp.append(nn.Linear(in_hid_out[-2], in_hid_out[-1], bias=False))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.mlp:
            x = layer(x)
        return x


class Train_my_NN():
    def __init__(self, model, train_data_loader, test_data_loader, loss_fn, optimizer, epochs):
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs

    def train_one_epoch(self):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(self.train_data_loader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
        return last_loss

    def test_one_epoch(self):
        running_loss = 0.
        for i, data in enumerate(self.test_data_loader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            running_loss += loss.item()
        return running_loss/len(self.test_data_loader)

    def train_nn(self):
        train_loss = []
        test_loss = []
        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.epochs))
            train_loss.append(self.train_one_epoch())
            test_loss.append(self.test_one_epoch())
            print('  train loss: {}'.format(train_loss[-1]))
            print('  test loss: {}'.format(test_loss[-1]))
        print('Finished training')
        return train_loss, test_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


num_neuron = 128
num_hidden_layer = 3
learning_rate = 1e-1
nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
net = NeuralNetwork(nn_structure)
training_set = torch.load(
    './damage identification task/data/neural_nets/train_data.pt')
test_set = torch.load(
    './damage identification task/data/neural_nets/test_data.pt')
training_loader = DataLoader(training_set, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
train_my_nn = Train_my_NN(net, training_loader, test_loader,
                          loss_function, optimizer, 300)
train_loss, test_loss = train_my_nn.train_nn()
