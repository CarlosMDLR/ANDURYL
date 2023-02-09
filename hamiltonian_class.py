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
from priors import priors
from tqdm import tqdm


def transform(param, a, b):
    """
    Return the transformed parameters using the logit transform to map from (-inf,inf) to (a,b)
    It also returns the log determinant of the Jacobian of the transformation.
    """
    logit_1 = 1.0 / (1.0+ torch.exp(-param))
    transformed = a + (b-a) * logit_1
    logdet = torch.log(torch.tensor((b-a))) + torch.log(logit_1) + torch.log((1.0 - logit_1))
    p=priors(A=a,B=b,N=1)
    prior = torch.log(torch.tensor(p.Uprior()))
    return transformed, logdet,prior

def invtransform(param,a,b):
    return(sp.logit((param-a)/(b-a)))

class hamiltonian_model:
    def __init__(self, image,psf,mask,sky_sigma,read_noise,gain,normfactor, model_type='Sersic'):
        self.image = image
        self.psf = psf
        self.mask=mask/np.max(mask)
        self.trues = np.where((self.mask !=1))
        self.trues_i=self.trues[0]
        self.trues_j=self.trues[1]
        self.nx, self.ny = self.image.shape
        self.sky_sigma=sky_sigma
        self.read_noise=read_noise
        self.normfactor=normfactor
        self.gain=gain
        self.model_class = profiles(y_size=self.image.shape[0],x_size=self.image.shape[1])

        self.model_type = model_type
        
        self.psf_mod = torch.tensor(self.psf)
        self.psf_fft = torch.fft.fft2(self.psf_mod)
        
        self.yt = torch.tensor(self.image)

        self.ny, self.nx = self.yt.shape

    def conv2d_fft_psf(self, f):
        ff = torch.fft.fft2(f)
        return torch.fft.fftshift(torch.fft.ifft2(ff * self.psf_fft)).real

    def hamiltonian_sersic(self):
        
        def logPosteriorSersic(pars):
            # pars[0]=amplitude
            # pars[1]= Re
            # pars[2]= n
            # pars[3]=x0
            # pars[4]=y0
            # pars[5]=ellip
            # pars[6]=theta            
            
            a0,lodget0,prior0 = transform(pars[0],0,1)
            a1,lodget1,prior1 = transform(pars[1],0.001,400)
            a2,lodget2,prior2 = transform(pars[2],0.001,10)
            a3,lodget3,prior3 = transform(pars[3],0,self.nx+1)
            a4,lodget4,prior4 = transform(pars[4],0,self.ny+1)
            a5,lodget5,prior5 = transform(pars[5],0,0.999)
            a6,lodget6,prior6 = transform(pars[6],0,180)

            if (self.model_type == 'Sersic'):
                model_method = self.model_class.Sersic(amp_sersic=a0*self.normfactor,r_eff_sersic=a1,\
                                       n_sersic=a2,x0_sersic=a3,\
                                           y0_sersic=a4,\
                                               ellip_sersic=a5,\
                                                   theta_sersic=a6)
            # modelin= conv2d_pyt(model_method, self.psf_mod)
            modelin= self.conv2d_fft_psf(model_method)/self.normfactor
            #breakpoint() 
            noise = torch.sqrt((modelin*self.normfactor+self.sky_sigma*self.gain +self.read_noise**2 ))/np.sqrt(self.normfactor)
            logL = (-0.5 * torch.sum(((modelin[self.trues_i[:],self.trues_j[:]] - self.yt[self.trues_i[:],self.trues_j[:]])**2) / (noise[self.trues_i[:],self.trues_j[:]]**2)))+lodget0.sum()+lodget1.sum()\
                +lodget2.sum()+lodget3.sum()+lodget4.sum()+lodget5.sum()+lodget6.sum()\
                    #+prior0+prior1+prior2+prior3+prior4+prior5+prior6
            
            N = np.shape(modelin[self.trues_i[:],self.trues_j[:]])[0]
            chi_2 =(1/N)* torch.sum(((modelin[self.trues_i[:],self.trues_j[:]] - self.yt[self.trues_i[:],self.trues_j[:]])**2) / (noise[self.trues_i[:],self.trues_j[:]]**2))
            print("chi^2= %.5f"%(chi_2))
            return logL

        #########################################################
        # # Test a normal fit to the observations using gradient descent

        # # Start from the center of the volume
        # paramis_transformed = np.zeros(7)                
        # paramis_init = torch.tensor(paramis_transformed,requires_grad=True)

        # # Optimize the parameters using Adam
        # optimizer = torch.optim.Adam([paramis_init], lr=0.1)

        # pbar = tqdm(total=2000)
        # for i in range(2000):
        #     optimizer.zero_grad()
        #     loss = -logPosteriorSersic(paramis_init)
        #     loss.backward()
        #     optimizer.step()
        #     pbar.set_postfix(loss=loss.item())
        #     pbar.update()

        # initial = torch.zeros(7)
        # initial[0], _ = transform(paramis_init[0],0,1)
        # initial[1], _ = transform(paramis_init[1],0.001,400)
        # initial[2], _ = transform(paramis_init[2],0.001,10)
        # initial[3], _ = transform(paramis_init[3],0,self.nx)
        # initial[4], _ = transform(paramis_init[4],0,self.ny)
        # initial[5], _ = transform(paramis_init[5],0,0.999)
        # initial[6], _ = transform(paramis_init[6],0,180)

        # if (self.model_type == 'Sersic'):
        #     model_method = self.model_class.Sersic(amp_sersic=initial[0],r_eff_sersic=initial[1],\
        #                             n_sersic=initial[2],x0_sersic=initial[3],\
        #                                 y0_sersic=initial[4],\
        #                                     ellip_sersic=initial[5],\
        #                                         theta_sersic=initial[6])
        # modelin= self.conv2d_fft_psf(model_method).detach().numpy()

        # fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(10, 5))
        # ax[0].imshow(self.yt)
        # ax[1].imshow(modelin)
        # print(" ")
        # print('Optimized parameters: ', initial)
        # #breakpoint()


        #########################################################

        #hamiltorch.set_random_seed(123)        
        #paramis = np.array([24.51/3922.3203,80/0.396,5.10,154.94841,182.67604,1- 0.7658242620261781,84.6835058750300078])
        paramis = np.array([0.3,100,0.5,120,110,0.4,30])
        #paramis = np.array([0.1,1,1,1,1,0.1,0.1])
        
        paramis[0] = invtransform(paramis[0],0,1-1e-2)
        paramis[1] = invtransform(paramis[1],0.001,400-1)
        paramis[2] = invtransform(paramis[2],0.001,10-1e-1)
        paramis[3] = invtransform(paramis[3],0,self.nx+1)
        paramis[4] = invtransform(paramis[4],0,self.ny+1)
        paramis[5] = invtransform(paramis[5],0,0.999-1e-2)
        paramis[6] = invtransform(paramis[6],0,180-1)
        
        paramis_init = torch.tensor(paramis,requires_grad=True)
        
        burn = 500
        step_size = 100
        L =20
        N = 2000
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

        params_nuts[:, 0],_,_ = transform(params_nuts[:, 0],0,1-1e-2)
        params_nuts[:, 1],_,_ = transform(params_nuts[:, 1],0.001,400-1)
        params_nuts[:, 2],_,_ = transform(params_nuts[:, 2],0.001,10+1e-1)
        params_nuts[:, 3],_,_ = transform(params_nuts[:, 3],0,self.nx+1)
        params_nuts[:, 4],_,_ = transform(params_nuts[:, 4],0,self.ny+1)
        params_nuts[:, 5],_,_ = transform(params_nuts[:, 5],0,0.999-1e-2)
        params_nuts[:, 6],_,_ = transform(params_nuts[:, 6],0,180-1)

        #params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1).numpy()
        return(params_nuts,burn,step_size,L,N)
    
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