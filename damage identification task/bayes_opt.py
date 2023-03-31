import gpytorch
import torch
import math
from matplotlib import pyplot as plt
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BayesOpt():
    def __init__(self, train_x, train_y, test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x

    def train_gp(self, training_iter = 50):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.train_x, self.train_y, likelihood)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
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
        return observed_pred.mean.numpy(), lower.numpy(), upper.numpy()
    
    def find_next(self):
        _, lower, _ = self.predict()
        next_x = self.test_x[np.argmin(lower)]
        return next_x

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
    