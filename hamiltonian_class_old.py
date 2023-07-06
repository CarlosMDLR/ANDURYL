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
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

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
    """
    Return the inverse of the transformed parameters
    """
    return(sp.logit((param-a)/(b-a)))

class hamiltonian_model:
    """
    Class where the different adjustment methods are grouped according to the different functional forms
    
    Implemented functions:
        -Sersic (in progress)
        -Exponential (in progress)
        -Ferrers (in progress)
    """
    def __init__(self, image,psf,mask,sky_sigma,read_noise,gain, normfactor,model_type='Sersic'):

        # Here the mask is applied keeping the indices that have not been omitted
        self.mask=mask/np.max(mask)
        self.trues = np.where((self.mask !=1))
        self.trues_i=self.trues[0]
        self.trues_j=self.trues[1]
        #---------------------------------------------------------------------#
        self.image = image #image of the galaxy to adjust
        self.psf = psf #psf previously generated in the main program 
        self.sky_sigma=sky_sigma
        self.read_noise=read_noise
        self.gain=gain
        self.model_class = profiles(y_size=self.image.shape[0],x_size=self.image.shape[1])
        self.normfactor=normfactor
        self.model_type = model_type
        #Here we proceed to carry out previous steps for the convolution with the psf
        #You have to do zero padding and be careful where python starts to perform the convolution
        #That is why they are made iffftshift now
        self.sz = (self.image.shape[0] - self.psf.shape[0], self.image.shape[1] - self.psf.shape[1])
        self.kernel = np.pad(self.psf, (((self.sz[0]+1)//2, self.sz[0]//2), ((self.sz[1]+1)//2, self.sz[1]//2)))
        self.psf_mod = torch.tensor(self.kernel)
        self.psf_fft = torch.fft.fft2( torch.fft.ifftshift(self.psf_mod))
        
        self.yt = torch.tensor(self.image)

        self.ny, self.nx = self.yt.shape # shape of the galaxy image

    def conv2d_fft_psf(self, f):
        """
        Function that performs the convolution with the psf
        """
        ff = torch.fft.fft2(f)
        return (torch.fft.ifft2(ff * self.psf_fft)).real

    def hamiltonian_sersic(self):
        """
        Function to adjust the galaxy with the Sersic profile
        """
        def logPosteriorSersic(pars):
            #List of params
            # pars[0]=amplitude
            # pars[1]= Re
            # pars[2]= n
            # pars[3]=x0
            # pars[4]=y0
            # pars[5]=ellip
            # pars[6]=theta            
            
            #Here a transformation to parameter space is made, in order 
            #to generate the model of the galaxy in physical units.
            #In the parameters of the center, the entire image is not chosen 
            #so that there are no conflicts with the oversampling and the edges of the image
            a0,lodget0,prior0 = transform(pars[0],0,1)
            a1,lodget1,prior1 = transform(pars[1],0.001,200)
            a2,lodget2,prior2 = transform(pars[2],0.001,10)
            a3,lodget3,prior3 = transform(pars[3],15,self.nx-15)
            a4,lodget4,prior4 = transform(pars[4],15,self.ny-15)
            a5,lodget5,prior5 = transform(pars[5],0,0.999)
            a6,lodget6,prior6 = transform(pars[6],0,180)
            
            #The model is created using a Sersic profile
            if (self.model_type == 'Sersic'):
                model_method = self.model_class.Sersic(amp_sersic=a0,r_eff_sersic=a1,\
                                       n_sersic=a2,x0_sersic=a3,\
                                           y0_sersic=a4,\
                                               ellip_sersic=a5,\
                                                   theta_sersic=a6)
            #Convolution with the PSF
            modelin= self.conv2d_fft_psf(model_method)
            
            #The noise and logL are calculated and the latter is returned.
            noise = torch.sqrt((abs(self.yt)*self.normfactor+self.sky_sigma*self.gain +self.read_noise**2 ))/np.sqrt(self.normfactor)
            logL = (-0.5 * torch.sum(((abs(self.yt[self.trues_i[:],self.trues_j[:]])-modelin[self.trues_i[:],self.trues_j[:]])**2) / (noise[self.trues_i[:],self.trues_j[:]]**2)))+lodget0.sum()+lodget1.sum()\
                +lodget2.sum()+lodget3.sum()+lodget4.sum()+lodget5.sum()+lodget6.sum()\
                    #+prior0+prior1+prior2+prior3+prior4+prior5+prior6
            
            N = np.shape(modelin[self.trues_i[:],self.trues_j[:]])[0]
            chi_2 =(1/N)* torch.sum(((abs(self.yt[self.trues_i[:],self.trues_j[:]])-modelin[self.trues_i[:],self.trues_j[:]])**2) / (noise[self.trues_i[:],self.trues_j[:]]**2))
            print("chi^2= %.5f"%(chi_2))
            return logL
      
        #paramis = np.array([24.51/3922.3203,80,5.10,182.67604,154.94841,1- 0.7658242620261781,84.6835058750300078])
        #Initial parameters
        paramis = np.array([0.3,100,4.5,150,150,0.4,30])
        
        paramis[0] = invtransform(paramis[0],0,1)
        paramis[1] = invtransform(paramis[1],0.001,200)
        paramis[2] = invtransform(paramis[2],0.001,10)
        paramis[3] = invtransform(paramis[3],15,self.nx-15)
        paramis[4] = invtransform(paramis[4],15,self.ny-15)
        paramis[5] = invtransform(paramis[5],0,0.999)
        paramis[6] = invtransform(paramis[6],0,180)
        
        paramis_init = torch.tensor(paramis,requires_grad=True)
        
        #Various hamiltorch parameters
        burn = 500 #Number of initial iterations with a step size equal to the position below
        step_size = 10 #initial step_size
        L =10 #Number of "jumps" per iteration
        N = 1500 #Number of iterations with an adjusted step after the burn phase
        N_nuts = burn + N
        params_nuts = hamiltorch.sample(log_prob_func=logPosteriorSersic, 
                                        params_init=paramis_init, 
                                        num_samples=N_nuts,
                                        step_size=step_size, 
                                        num_steps_per_sample=L,
                                        sampler=hamiltorch.Sampler.HMC_NUTS,
                                        burn=burn,
                                        desired_accept_rate=0.8)
        
        # logPost=[]
        # for j in range(0,len(params_nuts)):
        #     a = logPosteriorSersic(params_nuts[j])
        #     logPost.append(a)
        
        params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1)

        params_nuts[:, 0],_,_ = transform(params_nuts[:, 0],0,1)
        params_nuts[:, 1],_,_ = transform(params_nuts[:, 1],0.001,200)
        params_nuts[:, 2],_,_ = transform(params_nuts[:, 2],0.001,10)
        params_nuts[:, 3],_,_ = transform(params_nuts[:, 3],15,self.nx-15)
        params_nuts[:, 4],_,_ = transform(params_nuts[:, 4],15,self.ny-15)
        params_nuts[:, 5],_,_ = transform(params_nuts[:, 5],0,0.999)
        params_nuts[:, 6],_,_ = transform(params_nuts[:, 6],0,180)
        
        #max_logpost=np.argmax(logPost)
        #max_logpost_params= params_nuts[max_logpost]
        #params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1).numpy()
        return(params_nuts,burn,step_size,L,N)#,max_logpost_params)
    
    # def hamiltonian_exponential(self):
    #     sigman = 1e-3
    #     y = self.image
    #     yt = torch.tensor(y,requires_grad=True)
    #     def logPosteriorSersic(pars):
    #         #pars[0]=amplitude
    #         #pars[1]= Re
    #         #pars[2]= n
    #         #pars[3]=x0
    #         #pars[4]=y0
    #         #pars[5]=ellip
    #         #pars[6]=theta
    #         model_class = profiles(x_size=y.shape[0],y_size=y.shape[1],\
    #                                amp_exp=pars[0],h_exp=pars[1],x0_exp=pars[2],\
    #                                        y0_exp=pars[3],\
    #                                            ellip_exp=pars[4],\
    #                                                theta_exp=pars[5])
    #         model_method = getattr(model_class, 'Exponential')
    #         logL = -0.5 * torch.sum((model_method() - yt)**2 / sigman**2)
    #         return logL
    #     #hamiltorch.set_random_seed(123)
    #     paramis = np.array([24.51,80,154.94841,182.67604,0.765,5])
    #     params_init = torch.tensor(paramis,requires_grad=True)
    #     burn = 500
    #     step_size = 0.1
    #     L = 5
    #     N = 1000
    #     N_nuts = burn + N
    #     params_nuts = hamiltorch.sample(log_prob_func=logPosteriorSersic, 
    #                                     params_init=params_init, 
    #                                     num_samples=N_nuts,
    #                                     step_size=step_size, 
    #                                     num_steps_per_sample=L,
    #                                     sampler=hamiltorch.Sampler.HMC_NUTS,
    #                                     burn=burn,
    #                                     desired_accept_rate=0.8)
    #     params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1)
    #     return(params_nuts)
    
    # def hamiltonian_ferrers(self):
    #     sigman = 1e-3
    #     y = self.image
    #     yt = torch.tensor(y,requires_grad=True)
    #     def logPosteriorSersic(pars):
    #         #pars[0]=amplitude
    #         #pars[1]= Re
    #         #pars[2]= n
    #         #pars[3]=x0
    #         #pars[4]=y0
    #         #pars[5]=ellip
    #         #pars[6]=theta
    #         model_class = profiles(x_size=y.shape[0],y_size=y.shape[1],\
    #                                amp_ferrers=pars[0],a_bar_ferrers=pars[1],\
    #                                    n_bar_ferrers=pars[2],x0_ferrers=pars[3],\
    #                                        y0_ferrers=pars[4],\
    #                                            ellip_ferrers=pars[5],\
    #                                                theta_ferrers=pars[6])
    #         model_method = getattr(model_class, 'Ferrers')
    #         logL = -0.5 * torch.sum((model_method() - yt)**2 / sigman**2)
    #         return logL
    #     #hamiltorch.set_random_seed(123)
    #     paramis = np.array([24.51,80,5.10,154.94841,182.67604,0.765,5])
    #     params_init = torch.tensor(paramis,requires_grad=True)
    #     burn = 500
    #     step_size = 0.1
    #     L = 5
    #     N = 1000
    #     N_nuts = burn + N
    #     params_nuts = hamiltorch.sample(log_prob_func=logPosteriorSersic, 
    #                                     params_init=params_init, 
    #                                     num_samples=N_nuts,
    #                                     step_size=step_size, 
    #                                     num_steps_per_sample=L,
    #                                     sampler=hamiltorch.Sampler.HMC_NUTS,
    #                                     burn=burn,
    #                                     desired_accept_rate=0.8)
    #     params_nuts = torch.cat(params_nuts[1:]).reshape(len(params_nuts[1:]),-1)
    #     return(params_nuts)