# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:26:50 2022

@author: Usuario
"""

import torch
import numpy as np
import matplotlib.pyplot as pl
import hamiltorch

# Bayesian inference with flat priors for a parabola

# Let us first define the parabola with n points and a certain
# amount of noise
n = 15
sigman = 1e-1
truth = [0.3, -0.2, 0.1]
x = np.linspace(-3, 3, 15)
y = truth[0] * x**2 + truth[1] * x + truth[2]
ynoise = y + np.random.normal(loc=0, scale=sigman, size=n)


# Transform the numpy arrays into pytorch tensors
# One can also do it directly with pytorch
xt = torch.tensor(x)
yt = torch.tensor(ynoise)

# Function that returns the logPosterior
# We evaluate the model and assume Gaussian noise with zero mean and 
# standard deviation sigman
def logPosterior(pars):
    model = pars[0] * xt**2 + pars[1] * xt + pars[2]
    logL = -0.5 * torch.sum((model - yt)**2 / sigman**2)
    return logL

# Sampler
# NUTS
#hamiltorch.set_random_seed(123)
params_init = torch.ones(3)
burn = 500
step_size = 0.3
L = 5
N = 1000
N_nuts = burn + N
params_nuts = hamiltorch.sample(log_prob_func=logPosterior, 
                                params_init=params_init, 
                                num_samples=N_nuts,
                                step_size=step_size, 
                                num_steps_per_sample=L,
                                sampler=hamiltorch.Sampler.HMC_NUTS,
                                burn=burn,
                                desired_accept_rate=0.8)

# Concatenate the list of points that is returned into a pytorch tensor and to a numpy array
params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1).numpy()

# Do some plots
fig, ax = pl.subplots(nrows=2, ncols=2)
ax[0, 0].plot(x, ynoise, 'o')
for i in range(50):
    pars = params_nuts[i, :]
    ax[0, 0].plot(x, pars[0] * x**2 + pars[1] * x + pars[2], alpha=0.2, color='C1')

for i in range(3):
    ax.flat[1+i].hist(params_nuts[:, i], bins=30)
    ax.flat[1+i].axvline(truth[i], color='C1')

pl.show()