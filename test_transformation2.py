import torch
import hamiltorch
import matplotlib.pyplot as pl
import scipy.special as sp
import numpy as np

left = 0.0
right = 5.0

support = torch.distributions.constraints.interval(left, right)

def transform(param,a,b):
    return( a + (b-a) / (1.0+ torch.exp(-param)))

def transform_np(param,a,b):
    logit_1 = 1.0 / (1.0+ np.exp(-param))
    transformed = a + (b-a) * logit_1
    logdet = np.log((b-a) * logit_1 * (1.0 - logit_1))
    return transformed, logdet

def logit(u):
    return np.log(u / (1.0-u))

def invtransform_np(param,a,b):
    return logit( (param-a)/(b-a) )


def log_prob_np(u):
    
    mean = torch.tensor([0.0, 0.0])
    stddev = torch.tensor([1., 1.]) 

    dist = torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2))

    constrained, logdet = transform_np(u, left, right)

    constrained = torch.tensor(constrained, dtype=torch.float32)
    
    log_prob = dist.log_prob(constrained).sum().numpy() + logdet.sum()
            
    return log_prob

N = 400
step_size = 0.3
L = 5

hamiltorch.set_random_seed(123)

params_init_np = 0.5 * (right + left) * np.ones(2)
params_init = torch.tensor(params_init_np.astype('float64'))

burn=100
N_nuts = burn + N

import emcee

pos = params_init_np + 1e-2 * np.random.randn(32, 2)

nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_prob_np)
sampler.run_mcmc(pos, 500, progress=True)
samples = sampler.get_chain()[200:, :, :]
samples = samples.reshape((-1, ndim))

samples2, _ = transform_np(samples, left, right)

pl.plot(samples2[:,0].flatten(),samples2[:,1].flatten(),'.')


# params_hmc_nuts = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init, num_samples=N,
                            #    step_size=step_size, num_steps_per_sample=L)

# params_hmc_nuts = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init,
#                                                   num_samples=N_nuts, step_size=step_size, num_steps_per_sample=L,
#                                                   sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
#                                                   desired_accept_rate=0.8)

# # params_hmc_nuts = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init, num_samples=N,
#                                 #   step_size=step_size,num_steps_per_sample=L, sampler=hamiltorch.Sampler.RMHMC,
#                                 #   integrator=hamiltorch.Integrator.IMPLICIT, fixed_point_max_iterations=1000,
#                                 #   fixed_point_threshold=1e-05)

# coords_nuts = torch.cat(params_hmc_nuts).reshape(len(params_hmc_nuts),-1)

# coords_nuts = torch.distributions.biject_to(support)(coords_nuts)



# tmp = coords_nuts.numpy()

# fig, ax = pl.subplots()
# ax.plot(tmp[:, 0], tmp[:, 1], '.')
# pl.show()

# print(np.std(tmp, axis=0))