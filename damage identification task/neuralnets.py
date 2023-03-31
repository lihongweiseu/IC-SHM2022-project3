import numpy as np
import torch
import torch.nn as nn
from fembeam import beam_fem
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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


def train_one_epoch(model, data_loader, loss_fn, optimizer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        inputs = inputs.float()
        labels = labels.to(device)
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss


def test_one_epoch(model, data_loader, loss_fn):
    running_loss = 0.
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        inputs = inputs.float()
        labels = labels.to(device)
        labels = labels.float()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
    return running_loss/len(data_loader)


def train_nn(model, train_data_loader, test_data_loader, loss_fn, optimizer, epochs):
    train_loss = []
    test_loss = []
    for t in range(epochs):
        print('Epoch {} of {}'.format(t + 1, epochs))
        model.train(True)
        avg_train_loss = train_one_epoch(
            model, train_data_loader, loss_fn, optimizer)
        model.train(False)
        train_loss.append(avg_train_loss)
        avg_test_loss = test_one_epoch(model, test_data_loader, loss_fn)
        test_loss.append(avg_test_loss)
        print('  train loss: {}'.format(avg_train_loss))
        print('  test loss: {}'.format(avg_test_loss))
    print('Finished training')


training_set = torch.load(
    './damage identification task/data/neural_nets/train_data.pt')
test_set = torch.load(
    './damage identification task/data/neural_nets/test_data.pt')
training_loader = DataLoader(training_set, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
net = NeuralNetwork([4, 128, 128, 128, 3])
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
loss_function = nn.MSELoss()
train_nn(net, training_loader, test_loader, loss_function, optimizer, 3000)
