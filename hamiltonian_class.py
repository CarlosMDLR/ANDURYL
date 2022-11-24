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
        sigman = 1
        y = self.image
        yt = torch.tensor(y)
        def logPosteriorSersic(pars):
            #pars[0]=amplitude
            #pars[1]= Re
            #pars[2]= n
            #pars[3]=x0
            #pars[4]=y0
            #pars[5]=ellip
            #pars[6]=theta
            model_class = profiles(x_size=np.shape(y)[0],y_size=np.shape(y)[0],\
                                   amp_sersic=pars[0],r_eff_sersic=pars[1],\
                                       n_sersic=pars[2],x0_sersic=pars[3],\
                                           y0_sersic=pars[4],\
                                               ellip_sersic=pars[5],\
                                                   theta_sersic=pars[6])
            model_method = getattr(model_class, 'Sersic')
            model = model_method
            logL = -0.5 * torch.sum((model - yt)**2 / sigman**2)
            return logL
        hamiltorch.set_random_seed(123)
        paramis = np.array([24.51,5,5.10,182.67604,154.94841,0.765,0.4])
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