import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
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
                nn.Linear(in_hid_out[i], in_hid_out[i+1], bias=True))
            self.mlp.append(nn.Tanh())
        self.mlp.append(nn.Linear(in_hid_out[-2], in_hid_out[-1], bias=True))

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
        avg_cumu_loss = 0.
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
            avg_cumu_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
        return avg_cumu_loss/len(self.train_data_loader)

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
    def __init__(self, num_neuron, num_hidden_layer, learning_rate, data_path, use_norm=True):
        self.num_neuron = num_neuron
        self.num_hidden_layer = num_hidden_layer
        self.learning_rate = learning_rate
        self.data_path = data_path
        self.nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
        self.net = NeuralNetwork(self.nn_structure)
        self.use_norm = use_norm

    def train_one_nn(self, epochs=1):
        if self.use_norm:
            training_set = torch.load(self.data_path + 'train_data.pt')
            test_set = torch.load(self.data_path + 'test_data.pt')
        else:
            training_set = torch.load(
                self.data_path + 'train_data_without_norm.pt')
            test_set = torch.load(self.data_path + 'test_data_without_norm.pt')
        training_loader = DataLoader(
            training_set, batch_size=1000, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.learning_rate)
        train_my_nn = Train_my_NN(self.net, training_loader, test_loader,
                                  loss_function, optimizer, epochs)
        train_loss, test_loss = train_my_nn.train_nn()
        return train_loss, test_loss

    def save_one_nn(self):
        torch.save(self.net.state_dict(), self.data_path + 'nn_model.pt')

    def save_loss(self, train_loss, test_loss):
        np.save(self.data_path + 'train_loss.npy', train_loss)
        np.save(self.data_path + 'test_loss.npy', test_loss)


class Establish_nn_bayes_opt():
    def __init__(self, data_path, num_max_iter=50, use_norm=True):
        self.data_path = data_path
        self.num_max_iter = num_max_iter
        self.use_norm = use_norm

    def obj_fun(self, x):
        # all the parameters are normalized to [0,1]
        # x: a list of three parameters
        # will return the log10 of the loss after 10 epochs
        num_neuron, num_hidden_layer, learning_rate = x
        # tran_num_neuron = int(num_neuron*255+1)
        # tran_num_hidden_layer = int(num_hidden_layer*7+1)
        # tran_learning_rate = 10.0**(learning_rate*-5)
        nn_model = Establish_nn(num_neuron, num_hidden_layer,
                                learning_rate, self.data_path, self.use_norm)
        _, test_loss = nn_model.train_one_nn(40)
        return np.log10(test_loss[-1])

    def iter_bayes_opt(self):
        space = [Integer(10, 385, name='num_neuron', transform='normalize'),
                 Integer(1, 8, name='num_hidden_layer', transform='normalize'),
                 Real(1e-4, 1e-1, name='learning_rate', prior='log-uniform', transform='normalize')]
        res = gp_minimize(self.obj_fun,  # the function to minimize
                          # the bounds on each dimension of x
                          space,
                          acq_func="EI",  # the acquisition function
                          n_calls=self.num_max_iter,  # the number of evaluations of f
                          n_random_starts=5,  # the number of random initialization points
                          random_state=1111,  # the random seed
                          verbose=True)
        return res

    def save_hist(self, res):
        dump(res, self.data_path+'hist.pkl')

    def plot_cvg(self, res):
        plot_convergence(res)
        plt.show()


def bayes_opt_nn(data_path, num_max_iter, use_norm):
    nn_bayes_opt = Establish_nn_bayes_opt(
        data_path, num_max_iter, use_norm)
    res = nn_bayes_opt.iter_bayes_opt()
    nn_bayes_opt.save_hist(res)
    nn_bayes_opt.plot_cvg(res)


def use_res_train_nn(data_path, use_norm):
    res = load(data_path+'hist.pkl')
    num_neuron, num_hidden_layer, learning_rate = res.x
    # tran_num_neuron = int(num_neuron*255+1)
    # tran_num_hidden_layer = int(num_hidden_layer*7+1)
    # tran_learning_rate = 10.0**(learning_rate*-5)
    nn_model = Establish_nn(num_neuron, num_hidden_layer,
                            learning_rate, data_path, use_norm)
    train_loss, test_loss = nn_model.train_one_nn(200)
    nn_model.save_one_nn()
    nn_model.save_loss(train_loss, test_loss)


# if __name__ == '__main__':
    # data_path = r'./damage identification task/data/neural_nets/'
    # if you want to try the bayes optimization, uncomment the following line
    # bayes_opt_nn(data_path, num_max_iter=80, use_norm=True)
    # train the nn with the best parameters
    # use_res_train_nn(data_path, use_norm=True)
