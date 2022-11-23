# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:18:01 2022

@author: Usuario
"""

import torch
import numpy as np
import matplotlib.pyplot as pl
import hamiltorch



class hamiltonian_model:
    def __init__(self,image):
        self.image = image
    def hamiltonian_sersic(self):
        sigman = 1
        y = self.image
        x = np.linspace(0,376,376)
        xt = torch.tensor(x)
        yt = torch.tensor(y)
        def logPosteriorSersic(pars):
            #pars[0]=amplitude
            #pars[1]= bn
            #pars[2]= Re
            #pars[3]=n
            model = pars[0] * torch.exp(-pars[1] * ((xt/pars[2]) ** (1 / pars[3]) - 1))
            logL = -0.5 * torch.sum((model - yt)**2 / sigman**2)
            return logL
        hamiltorch.set_random_seed(123)
        paramis = np.array([24.51,0.765,5.,5.103])
        params_init = torch.tensor(paramis)
        burn = 500
        step_size = 0.1
        L = 5
        N = 1000
        N_nuts = burn + N
        params_nuts = hamiltorch.sample(log_prob_func=logPosteriorSersic, 
                                        params_init=params_init, 
                                        num_samples=N_nuts,
                                        step_size=step_size, 
                                        num_steps_per_sample=L,
                                        sampler=hamiltorch.Sampler.HMC_NUTS,
                                        burn=burn,
                                        desired_accept_rate=0.8)
        params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1).numpy()
        return(params_nuts)