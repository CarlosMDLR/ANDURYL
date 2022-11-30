# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:18:01 2022

@author: Usuario
"""

import torch
import numpy as np
import matplotlib.pyplot as pl
import hamiltorch
from profile_select import *

class hamiltonian_model:
    def __init__(self,image):
        self.image = image
    def hamiltonian_sersic(self):
        sigman = 1e-3
        y = self.image
        yt = torch.tensor(y,requires_grad=True)
        def logPosteriorSersic(pars):
            #pars[0]=amplitude
            #pars[1]= Re
            #pars[2]= n
            #pars[3]=x0
            #pars[4]=y0
            #pars[5]=ellip
            #pars[6]=theta
            model_class = profiles(x_size=y.shape[0],y_size=y.shape[1],\
                                   amp_sersic=pars[0].detach().numpy(),r_eff_sersic=pars[1].detach().numpy(),\
                                       n_sersic=pars[2].detach().numpy(),x0_sersic=pars[3].detach().numpy(),\
                                           y0_sersic=pars[4].detach().numpy(),\
                                               ellip_sersic=pars[5].detach().numpy(),\
                                                   theta_sersic=pars[6].detach().numpy())
            model_method = getattr(model_class, 'Sersic')
            m= model_method()
            model = torch.tensor(m,requires_grad=True)
            logL = -0.5 * torch.sum((model - yt)**2 / sigman**2)
            print(logL)
            return logL
        #hamiltorch.set_random_seed(123)
        paramis = np.array([24.51,80,5.10,154.94841,182.67604,0.765,5])
        params_init = torch.tensor(paramis,requires_grad=True)
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