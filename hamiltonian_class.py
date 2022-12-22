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
import torch.nn.functional as F
from profile_select import *


def transform(param, a, b):
    """
    Return the transformed parameters using the logit transform to map from (-inf,inf) to (a,b)
    It also returns the log determinant of the Jacobian of the transformation.
    """
    logit_1 = 1.0 / (1.0+ torch.exp(-param))
    transformed = a + (b-a) * logit_1
    logdet = torch.log(torch.tensor((b-a))) + torch.log(logit_1) + torch.log((1.0 - logit_1))
    return transformed, logdet

def invtransform(param,a,b):
    # return( a + (b-a) / (1.0+ torch.exp(-param)))
    return(sp.logit((param-a)/(b-a)))

# Too slow...
def conv2d_pyt(f, g):
    assert len(f.size()) == 2
    assert len(g.size()) == 2

    f_new = f.unsqueeze(0).unsqueeze(0)
    g_new = g.unsqueeze(0).unsqueeze(0)

    pad_y = (g.size(0) - 1) // 2
    pad_x = (g.size(1) - 1) // 2

    fcg = F.conv2d(f_new, g_new, bias=None, padding=(pad_y, pad_x))
    return fcg[0, 0, :, :]
def matmul_complex(t1,t2):
    return torch.view_as_complex(torch.stack((t1.real*t2.real - t1.imag*t2.imag, t1.real*t2.imag + t1.imag*t2.real),dim=2))
def conv2d_fft(f, g):

    size_y = f.size(0) + g.size(0) - 1
    size_x = f.size(1) + g.size(1) - 1

    f_new = torch.zeros((size_y, size_x))
    g_new = torch.zeros((size_y, size_x))

    # copy f to center
    f_pad_y = (f_new.size(0) - f.size(0)) // 2
    f_pad_x = (f_new.size(1) - f.size(1)) // 2
    f_new[f_pad_y:-f_pad_y, f_pad_x:-f_pad_x] = f

    # anchor of g is 0,0 (flip g and wrap circular)
    g_center_y = g.size(0) // 2
    g_center_x = g.size(1) // 2
    g_y, g_x = torch.meshgrid(torch.arange(g.size(0)), torch.arange(g.size(1)))
    g_new_y = (g_y.flip(0) - g_center_y) % g_new.size(0)
    g_new_x = (g_x.flip(1) - g_center_x) % g_new.size(1)
    g_new[g_new_y, g_new_x] = g[g_y, g_x].float()

    # take fft of both f and g
    F_f = torch.fft.rfft(f_new)
    F_g = torch.fft.rfft(g_new)
   
    # inverse fft
    fcg = torch.fft.irfft(matmul_complex(F_f,F_g))

    # crop center before returning
    return fcg[f_pad_y:-f_pad_y, f_pad_x:-f_pad_x]

class hamiltonian_model:
    def __init__(self, image,psf, model_type='Sersic', sigman=1e-3):
        self.image = image

        self.psf = psf

        self.model_class = profiles(x_size=self.image.shape[0],y_size=self.image.shape[1])

        self.model_type = model_type
        
        self.psf_mod = torch.tensor(self.psf)
        
        self.yt = torch.tensor(self.image)

        self.sigman = sigman * torch.max(self.yt)

    def hamiltonian_sersic(self):
        
        def logPosteriorSersic(pars):
            # pars[0]=amplitude
            # pars[1]= Re
            # pars[2]= n
            # pars[3]=x0
            # pars[4]=y0
            # pars[5]=ellip
            # pars[6]=theta            
            
            a0,lodget0 = transform(pars[0],0,1)
            a1,lodget1 = transform(pars[1],0.001,400)
            a2,lodget2 = transform(pars[2],0.001,10)
            a3,lodget3 = transform(pars[3],0,400)
            a4,lodget4 = transform(pars[4],0,400)
            a5,lodget5 = transform(pars[5],0,0.999)
            a6,lodget6 = transform(pars[6],0,180)

            if (self.model_type == 'Sersic'):
                model_method = self.model_class.Sersic(amp_sersic=a0,r_eff_sersic=a1,\
                                       n_sersic=a2,x0_sersic=a3,\
                                           y0_sersic=a4,\
                                               ellip_sersic=a5,\
                                                   theta_sersic=a6)
            modelin= conv2d_pyt(model_method, self.psf_mod)
            #breakpoint()  
            logL = (-0.5 * torch.sum(((modelin - self.yt)**2) / (self.sigman**2)))+lodget0.sum()+lodget1.sum()\
                +lodget2.sum()+lodget3.sum()+lodget4.sum()+lodget5.sum()+lodget6.sum()
           
            #breakpoint()
            return logL

        #hamiltorch.set_random_seed(123)        
        #paramis = np.array([24.51/3922.3203,80/0.396,5.10,154.94841,182.67604,1- 0.7658242620261781,84.6835058750300078])
        paramis = np.array([0.1,200,1,100,100,0.1,0.1])
        
        paramis[0] = invtransform(paramis[0],0,1)
        paramis[1] = invtransform(paramis[1],0.001,400)
        paramis[2] = invtransform(paramis[2],0.001,10)
        paramis[3] = invtransform(paramis[3],0,400)
        paramis[4] = invtransform(paramis[4],0,400)
        paramis[5] = invtransform(paramis[5],0,0.999)
        paramis[6] = invtransform(paramis[6],0,180)
        
        paramis_init = torch.tensor(paramis,requires_grad=True)
        
        burn = 10
        step_size = 0.1
        L = 5
        N = 110
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

        params_nuts[:, 0],_ = transform(params_nuts[:, 0],0,1)
        params_nuts[:, 1],_ = transform(params_nuts[:, 1],0.001,400)
        params_nuts[:, 2],_ = transform(params_nuts[:, 2],0.001,10)
        params_nuts[:, 3],_ = transform(params_nuts[:, 3],0,400)
        params_nuts[:, 4],_ = transform(params_nuts[:, 4],0,400)
        params_nuts[:, 5],_ = transform(params_nuts[:, 5],0,0.999)
        params_nuts[:, 6],_ = transform(params_nuts[:, 6],0,180)

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