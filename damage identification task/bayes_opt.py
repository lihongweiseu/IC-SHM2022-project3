import gpytorch
import torch
import math
from matplotlib import pyplot as plt
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims =3, active_dims = [0,1,2]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BayesOpt():
    def __init__(self, train_x, train_y, test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x

    def train_gp(self, training_iter = 300):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.train_x, self.train_y, likelihood)
        model.covar_module.base_kernel.lengthscale = torch.tensor([10, 1, 1])
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f lengthscales: %.6f, %.6f, %.6f  noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.squeeze()[0],
                model.covar_module.base_kernel.lengthscale.squeeze()[1],
                model.covar_module.base_kernel.lengthscale.squeeze()[2],
                model.likelihood.noise.item()
            ))
            optimizer.step()
        return model, likelihood
    
    def predict(self):
        model, likelihood = self.train_gp()
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(self.test_x))
        lower, upper = observed_pred.confidence_region()
        return observed_pred.mean.detach().numpy(), lower.detach().numpy(), upper.detach().numpy()
    
    def find_next(self):
        _, lower, _ = self.predict()
        self.lower = lower
        next_x = self.test_x[np.argmin(lower)]
        return next_x
    
    def plot_current_pred(self):
        lower = self.lower.reshape(33,7,21)
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        print(np.max(lower))
        print(np.min(lower))
        nums_neuron = np.linspace(10, 258, 33)
        nums_hidden_layer = np.linspace(2, 8, 7)
        nums_lr = np.linspace(-5, 0, 21)
        ax[0,0].plot(self.train_x.numpy()[:,0], self.train_x.numpy()[:,1], 'r*')
        ax[0,0].contourf(nums_neuron, nums_hidden_layer, np.sum(lower, axis = 2).T)
        ax[0,0].set_title('nums_neuron & nums_hidden_layer')

        ax[0,1].plot(self.train_x.numpy()[:,0], self.train_x.numpy()[:,2], 'r*')
        ax[0,1].contourf(nums_neuron, nums_lr, np.sum(lower, axis = 1).T)
        ax[0,1].set_title('nums_lr & nums_neuron')

        ax[1,0].plot(self.train_x.numpy()[:,1], self.train_x.numpy()[:,2], 'r*')
        ax[1,0].contourf(nums_hidden_layer, nums_lr, np.sum(lower, axis = 0).T)
        ax[1,0].set_title('nums_hidden_layer & nums_lr')
        plt.show()

# # Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 100)
# # True function is sin(2*pi*x) with Gaussian noise
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
# # initialize likelihood and model
# test_x = torch.linspace(0, 1, 51)
# bayes_opt = BayesOpt(train_x, train_y, test_x)
# a,b,c = bayes_opt.predict()
# plt.plot(train_x.numpy(), train_y.numpy(), 'k*')
# plt.plot(test_x.numpy(), a, 'b')
# plt.plot(test_x.numpy(),b)
# plt.plot(test_x.numpy(),c)
# plt.show()
# print(bayes_opt.find_next())
    