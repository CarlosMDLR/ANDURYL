# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:18:01 2022

@author: Usuario
"""

import torch
import numpy as np
import matplotlib.pyplot as pl
import hamiltorch
import scipy.special as sp
from profile_select import *

def transform(param,a,b):
    return( a + (b-a) / (1.0+ torch.exp(-param)))
    #return(torch.logit((param-a)/(b-a)))

def invtransform(param,a,b):
    # return( a + (b-a) / (1.0+ torch.exp(-param)))
    return(sp.logit((param-a)/(b-a)))

class hamiltonian_model:
    def __init__(self, image, model_type='Sersic', sigman=1e-3):
        self.image = image

        self.model_class = profiles(x_size=self.image.shape[0],y_size=self.image.shape[1])

        self.model_type = model_type

        self.yt = torch.tensor(self.image)

        self.sigman = sigman * torch.max(self.yt)

    def hamiltonian_sersic(self):
        
        def logPosteriorSersic(pars):
            #pars[0]=amplitude
            #pars[1]= Re
            #pars[2]= n
            #pars[3]=x0
            #pars[4]=y0
            #pars[5]=ellip
            #pars[6]=theta            
            
            a0 = transform(pars[0],0,60000)
            a1 = transform(pars[1],0.001,400)
            a2 = transform(pars[2],0.001,10)
            a3 = transform(pars[3],0,400)
            a4 = transform(pars[4],0,400)
            a5 = transform(pars[5],0,0.999)
            a6 = transform(pars[6],0,180)

            if (self.model_type == 'Sersic'):
                model_method = self.model_class.Sersic(amp_sersic=a0,r_eff_sersic=a1,\
                                       n_sersic=a2,x0_sersic=a3,\
                                           y0_sersic=a4,\
                                               ellip_sersic=a5,\
                                                   theta_sersic=a6)
            breakpoint()  
            logL = -0.5 * torch.sum((model_method - self.yt)**2 / self.sigman**2)

            return logL

        #hamiltorch.set_random_seed(123)        
        paramis = np.array([24.51,80/0.396,5.10,154.94841,182.67604,1-0.765,5])
        # paramis = np.array([12,12,1,100,100,0.1,0.1])
        paramis[0] = invtransform(paramis[0],0,60000)
        paramis[1] = invtransform(paramis[1],0.001,400)
        paramis[2] = invtransform(paramis[2],0.001,10)
        paramis[3] = invtransform(paramis[3],0,400)
        paramis[4] = invtransform(paramis[4],0,400)
        paramis[5] = invtransform(paramis[5],0,0.999)
        paramis[6] = invtransform(paramis[6],0,180)
        
        paramis_init = torch.tensor(paramis,requires_grad=True)
        
        burn = 200
        step_size = 0.1
        L = 5
        N = 500
        N_nuts = burn + N
        params_nuts = hamiltorch.sample(log_prob_func=logPosteriorSersic, 
                                        params_init=paramis_init, 
                                        num_samples=N_nuts,
                                        step_size=step_size, 
                                        num_steps_per_sample=L,
                                        sampler=hamiltorch.Sampler.HMC_NUTS,
                                        burn=burn,
                                        desired_accept_rate=0.8)
        params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1)

        params_nuts[:, 0] = transform(params_nuts[:, 0],0,60000)
        params_nuts[:, 1] = transform(params_nuts[:, 1],0.001,400)
        params_nuts[:, 2] = transform(params_nuts[:, 2],0.001,10)
        params_nuts[:, 3] = transform(params_nuts[:, 3],0,400)
        params_nuts[:, 4] = transform(params_nuts[:, 4],0,400)
        params_nuts[:, 5] = transform(params_nuts[:, 5],0,0.999)
        params_nuts[:, 6] = transform(params_nuts[:, 6],0,180)

        #params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1).numpy()
        return(params_nuts)
    
    def hamiltonian_exponential(self):
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
                                   amp_exp=pars[0],h_exp=pars[1],x0_exp=pars[2],\
                                           y0_exp=pars[3],\
                                               ellip_exp=pars[4],\
                                                   theta_exp=pars[5])
            model_method = getattr(model_class, 'Exponential')
            logL = -0.5 * torch.sum((model_method() - yt)**2 / sigman**2)
            return logL
        #hamiltorch.set_random_seed(123)
        paramis = np.array([24.51,80,154.94841,182.67604,0.765,5])
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
        params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1)
        return(params_nuts)
    
    def hamiltonian_ferrers(self):
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
                                   amp_ferrers=pars[0],a_bar_ferrers=pars[1],\
                                       n_bar_ferrers=pars[2],x0_ferrers=pars[3],\
                                           y0_ferrers=pars[4],\
                                               ellip_ferrers=pars[5],\
                                                   theta_ferrers=pars[6])
            model_method = getattr(model_class, 'Ferrers')
            logL = -0.5 * torch.sum((model_method() - yt)**2 / sigman**2)
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
        params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1)
        return(params_nuts)