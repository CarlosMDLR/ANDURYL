import torch
import hamiltorch
import matplotlib.pyplot as pl
import scipy.special as sp
import numpy as np

left = 0.0
right = 5.0

def transform(param, a, b):
    """
    Return the transformed parameters using the logit transform to map from (-inf,inf) to (a,b)
    It also returns the log determinant of the Jacobian of the transformation.
    """
    logit_1 = 1.0 / (1.0+ torch.exp(-param))
    transformed = a + (b-a) * logit_1
    logdet = torch.log(torch.tensor((b-a))) + torch.log(logit_1) + torch.log((1.0 - logit_1))
    return transformed, logdet

def log_prob(u):

    mean = torch.tensor([0.0, 0.0])
    stddev = torch.tensor([1., 1.]) 

    # Probability distribution
    dist = torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2))
        
    # Evaluate the constrained parameters and the log determinant of the Jacobian of the transformation
    constrained, logdet = transform(u, left, right)    
    
    # Compute the log probability and add the log determinant of the Jacobian of the transformation
    log_prob = dist.log_prob(constrained).sum() + logdet.sum()
            
    return log_prob

N = 1000
step_size = 0.01
L = 5

hamiltorch.set_random_seed(123)

params_init_np = 0.5 * (right + left) * np.ones(2)
params_init = torch.tensor(params_init_np.astype('float32'))

burn=100
N_nuts = burn + N

params_hmc_nuts = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init,
                                                  num_samples=N_nuts, step_size=step_size, num_steps_per_sample=L,
                                                  sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
                                                  desired_accept_rate=0.8)

# Transform the samples back to the original space
coords_nuts = torch.cat(params_hmc_nuts).reshape(len(params_hmc_nuts),-1)
coords_nuts, _ = transform(coords_nuts, left, right)

coords_nuts = coords_nuts.numpy()

fig, ax = pl.subplots()
ax.plot(coords_nuts[:, 0], coords_nuts[:, 1], '.')
pl.show()

print(np.std(tmp, axis=0))