import numpy as np
import torch
import torch.nn as nn
from fembeam import beam_fem
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from bayes_opt import BayesOpt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


class Establish_nn():
    def __init__(self, num_neuron, num_hidden_layer, learning_rate, data_path):
        self.num_neuron = num_neuron
        self.num_hidden_layer = num_hidden_layer
        self.learning_rate = learning_rate
        self.data_path = data_path
        self.nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
        self.net = NeuralNetwork(self.nn_structure)

    def train_one_nn(self):
        training_set = torch.load(self.data_path + 'train_data.pt')
        test_set = torch.load(self.data_path + 'test_data.pt')
        training_loader = DataLoader(
            training_set, batch_size=1000, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.learning_rate)
        train_my_nn = Train_my_NN(self.net, training_loader, test_loader,
                                  loss_function, optimizer, 1)
        train_loss, test_loss = train_my_nn.train_nn()
        return train_loss, test_loss


class establish_nn_bayes_opt():
    def __init__(self, hist_path, num_max_iter=50):
        self.hist_path = hist_path
        self.num_max_iter = num_max_iter

    def iter_bayes_opt(self):
        nums_neuron = np.linspace(10, 258, 33)
        nums_hidden_layer = np.linspace(2, 8, 7)
        nums_lr = np.linspace(-5, 0, 21)
        test_x = np.meshgrid(nums_neuron, nums_hidden_layer, nums_lr)
        test_x = np.array(test_x).reshape(3, -1).T
        train_x = []
        train_y = []
        for i in range(self.num_max_iter):
            if i == 0:
                nn_model = Establish_nn(128, 4, 1e-1, data_path)
            else:
                nn_model = Establish_nn(int(next_x[0]), int(
                    next_x[1]), 10**next_x[2], data_path)
            train_loss, test_loss = nn_model.train_one_nn()
            train_x.append([nn_model.num_neuron,
                           nn_model.num_hidden_layer, np.log10(nn_model.learning_rate)])
            train_y.append(np.log10(train_loss[-1]))
            bayesopt = BayesOpt(torch.FloatTensor(train_x), torch.FloatTensor(
                train_y), torch.FloatTensor(test_x))
            next_x = bayesopt.find_next()
            bayesopt.plot_current_pred()
            np.savetxt(self.hist_path +
                       'train_loss'+str(i)+'.txt', np.array(train_loss), delimiter=',')
            np.savetxt(self.hist_path +
                       'test_loss'+str(i)+'.txt', np.array(test_loss), delimiter=',')
            np.savetxt(self.hist_path +
                       'train_x'+str(i)+'.txt', np.array(train_x), delimiter=',')
            np.savetxt(self.hist_path +
                       'train_y'+str(i)+'.txt', np.array(train_y), delimiter=',')
        np.savetxt(self.hist_path + 'test_x.pt',
                   np.array(test_x), delimiter=',')


if __name__ == '__main__':
    data_path = './damage identification task/data/neural_nets/'
    hist_path = './damage identification task/data/neural_nets/hist/'
    nn_model = Establish_nn(128, 4, 1e-1, data_path)
    nn_bayes_opt = establish_nn_bayes_opt(hist_path, 30)
    nn_bayes_opt.iter_bayes_opt()
# num_neuron = 128
# num_hidden_layer = 3
# learning_rate = 1e-1
# nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
# net = NeuralNetwork(nn_structure)
# training_set = torch.load(
#     './damage identification task/data/neural_nets/train_data.pt')
# test_set = torch.load(
#     './damage identification task/data/neural_nets/test_data.pt')
# training_loader = DataLoader(training_set, batch_size=1000, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
# train_my_nn = Train_my_NN(net, training_loader, test_loader,
#                           loss_function, optimizer, 300)
# train_loss, test_loss = train_my_nn.train_nn()
