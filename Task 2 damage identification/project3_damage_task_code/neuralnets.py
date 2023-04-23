import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
from fembeam import beam_fem
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Gen_Data():
    def __init__(self, path):
        self.path = path

    def gen_train_data(self, grid_num=91, subtract_norm=True):
        # Generate training set
        alphas = np.linspace(0.1, 1, grid_num)
        alphas = np.array(np.meshgrid(alphas, alphas, alphas)
                          ).reshape(3, -1).transpose()
        alphas = np.ones_like(alphas)-alphas
        beam = beam_fem()
        md_1st_r = []
        ms_ratio_undam = beam.md1st_ratio()
        for i in range(grid_num**3):
            md_1st_r.append(beam.nn_input(
                alphas[i, :], ms_ratio_undam, subtract_norm))
            if i % (grid_num**2) == 0:
                print("%.2f" % ((i/(grid_num**3))*100), '%')
        md_1st_r = np.array(md_1st_r)
        train_data = TensorDataset(
            torch.tensor(md_1st_r), torch.tensor(alphas))
        if subtract_norm:
            torch.save(
                train_data, self.path+'train_data.pt')
        else:
            torch.save(
                train_data, self.path+'train_data_without_norm.pt')

    def gen_test_data(self, num=300000, subtract_norm=True):
        # Generate test set
        alphas_test = np.random.rand(num, 3)*0.9
        md_1st_r_test = []
        beam = beam_fem()
        ms_ratio_undam = beam.md1st_ratio()
        for i in range(num):
            md_1st_r_test.append(beam.nn_input(
                alphas_test[i, :], ms_ratio_undam, subtract_norm))
            if i % (num/10) == 0:
                print("%.2f" % ((i/num)*100), '%')
        md_1st_r_test = np.array(md_1st_r_test)
        test_data = TensorDataset(torch.tensor(
            md_1st_r_test), torch.tensor(alphas_test))
        if subtract_norm:
            torch.save(
                test_data, self.path+'test_data.pt')
        else:
            torch.save(
                test_data, self.path+'test_data_without_norm.pt')


class NeuralNetwork(nn.Module):
    def __init__(self, in_hid_out, act_fun='tanh'):
        # in_hid_out: a list of integers, the number of nodes for each layer
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.ModuleList()
        for i in range(len(in_hid_out)-2):
            self.mlp.append(
                nn.Linear(in_hid_out[i], in_hid_out[i+1], bias=True))
            if act_fun == 'tanh':
                self.mlp.append(nn.Tanh())
            elif act_fun == 'relu':
                self.mlp.append(nn.ReLU())
            elif act_fun == 'sigmoid':
                self.mlp.append(nn.Sigmoid())
        self.mlp.append(nn.Linear(in_hid_out[-2], in_hid_out[-1], bias=True))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.mlp:
            x = layer(x)
        return x


class Train_my_NN():
    def __init__(self, model, train_data_loader, test_data_loader, loss_fn, optimizer, batch_size, epochs):
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def train_one_epoch(self):
        # running_loss = 0.
        avg_cumu_loss = 0.
        # last_loss = 0.
        for i, data in enumerate(self.train_data_loader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            # running_loss += loss.item()
            avg_cumu_loss += loss.item()
            # if i % (self.batch_size/10) == (self.batch_size/10-1):
            #     last_loss = running_loss / \
            #         (self.batch_size/10)
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     running_loss = 0.
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
    def __init__(self, num_neuron, num_hidden_layer, learning_rate, batch_size, act_fun, data_path, use_norm=True):
        self.num_neuron = num_neuron
        self.num_hidden_layer = num_hidden_layer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.data_path = data_path
        self.nn_structure = [4]+[num_neuron]*num_hidden_layer+[3]
        self.net = NeuralNetwork(self.nn_structure, act_fun)
        self.use_norm = use_norm
        self.hyp_code = '_{}_{}_{}_{}_{}'.format(self.act_fun, str(self.num_neuron), str(self.num_hidden_layer), str(
            round(-np.log10(self.learning_rate), 1)), str(round(np.log10(self.batch_size), 1)))

    def train_one_nn(self, epochs=200):
        if self.use_norm:
            training_set = torch.load(self.data_path + 'train_data.pt')
            test_set = torch.load(self.data_path + 'test_data.pt')
        else:
            training_set = torch.load(
                self.data_path + 'train_data_without_norm.pt')
            test_set = torch.load(self.data_path + 'test_data_without_norm.pt')
        training_loader = DataLoader(
            training_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=True)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.learning_rate)
        train_my_nn = Train_my_NN(self.net, training_loader, test_loader,
                                  loss_function, optimizer, self.batch_size, epochs)
        train_loss, test_loss = train_my_nn.train_nn()
        return train_loss, test_loss

    def save_one_nn(self):
        torch.save(self.net.state_dict(), self.data_path +
                   'nn_model' + self.hyp_code + '.pt')

    def save_loss(self, train_loss, test_loss):
        np.save(self.data_path + 'train_loss' +
                self.hyp_code + '.npy', train_loss)
        np.save(self.data_path + 'test_loss' +
                self.hyp_code + '.npy', test_loss)


class Establish_nn_bayes_opt():
    def __init__(self, act_fun, data_path, num_max_iter=50, use_norm=True):
        self.act_fun = act_fun
        self.data_path = data_path
        self.num_max_iter = num_max_iter
        self.use_norm = use_norm

    def obj_fun(self, x):
        # all the parameters are normalized to [0,1]
        # x: a list of three parameters
        # will return the log10 of the loss after 10 epochs
        num_neuron, num_hidden_layer, learning_rate, batch_size = x
        # tran_num_neuron = int(num_neuron*255+1)
        # tran_num_hidden_layer = int(num_hidden_layer*7+1)
        # tran_learning_rate = 10.0**(learning_rate*-5)
        nn_model = Establish_nn(num_neuron, num_hidden_layer,
                                learning_rate, batch_size, self.act_fun, self.data_path, self.use_norm)
        _, test_loss = nn_model.train_one_nn(40)
        return np.log10(test_loss[-1])

    def iter_bayes_opt(self):
        space = [Integer(8, 512, name='num_neuron', prior='log-uniform', base=2, transform='normalize'),
                 Integer(1, 6, name='num_hidden_layer', transform='normalize'),
                 Real(1e-3, 1e-1, name='learning_rate',
                      prior='log-uniform', transform='normalize'),
                 Integer(1e1, 1e5, name='batch_size', prior='log-uniform', transform='normalize', dtype=int)]
        res = gp_minimize(self.obj_fun,  # the function to minimize
                          # the bounds on each dimension of x
                          space,
                          acq_func="EI",  # the acquisition function
                          n_calls=self.num_max_iter,  # the number of evaluations of f
                          n_random_starts=5,  # the number of random initialization points
                          random_state=1,  # the random seed
                          verbose=True)
        return res

    def save_hist(self, res):
        dump(res, self.data_path+'hist_' + self.act_fun + '.pkl')

    def plot_cvg(self, res):
        plot_convergence(res)
        plt.show()


def bayes_opt_nn(act_fun, data_path, num_max_iter, use_norm):
    nn_bayes_opt = Establish_nn_bayes_opt(
        act_fun, data_path, num_max_iter, use_norm)
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
    train_loss, test_loss = nn_model.train_one_nn(40)
    nn_model.save_one_nn()
    nn_model.save_loss(train_loss, test_loss)


# if __name__ == '__main__':
#     data_path = './Task 2 damage identification/project3_damage_task_code/data/neural_nets/'
    # Generate the training and test data based on finite element model
    # My_data = Gen_Data(data_path)
    # My_data.gen_train_data()
    # My_data.gen_test_data()

    # if you want to try the bayes optimization, uncomment the following line
    # bayes_opt_nn('tanh', data_path, num_max_iter=15, use_norm=True)
    # bayes_opt_nn('sigmoid', data_path, num_max_iter=15, use_norm=True)
    # bayes_opt_nn('relu', data_path, num_max_iter=15, use_norm=True)

    # train the nn with the best parameters
    # use_res_train_nn(data_path, use_norm=True)
