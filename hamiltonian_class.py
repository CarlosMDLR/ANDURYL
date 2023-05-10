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
from collections import OrderedDict
import dynesty
import nestle
import scipy.stats as stats

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
    def __init__(self, image,psf,mask,sky_sigma,read_noise,gain, normfactor, noise_map, model_type='Sersic'):

        # Here the mask is applied keeping the indices that have not been omitted
        if len(np.unique(mask))==1:
            self.mask = torch.tensor(1.0 - mask)
        else:
            self.mask = torch.tensor(1.0 - mask/np.max(mask))
        #---------------------------------------------------------------------#
        self.image = image #image of the galaxy to adjust
        self.psf = psf #psf previously generated in the main program 
        self.sky_sigma=sky_sigma
        self.read_noise=read_noise
        self.gain=gain
        self.noise_map = torch.tensor(noise_map)
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
    # Specialized functions to return the logL and chi2 only
    def logPosteriorSersic_logL(self, pars):
        with torch.no_grad():
            logL, chi2, model = self.logPosteriorSersic(pars, numpy_input=True)
        return logL
    
    def logPosteriorSersic_chi2(self, pars):
        logL, chi2, model = self.logPosteriorSersic(pars)
        return chi2
            
    def prior_transform_nested(self, u):
        x = np.array(u)
        for i, rv in enumerate(self.rv):
            x[i] = rv.ppf(u[i])
                
        return x
        # General function to return the logL and chi2
    def logPosteriorSersic(self, pars, numpy_input=False):
        #List of params
        # pars[0]=amplitude
        # pars[1]= Re
        # pars[2]= n
        # pars[3]=x0
        # pars[4]=y0
        # pars[5]=ellip
        # pars[6]=theta            
        
        if numpy_input:
            pars = torch.tensor(pars)

        a0 = 10.0**pars[0]
        a1 = 10.0**pars[1]
        a2 = pars[2]
        a3 = self.nx * pars[3]
        a4 = self.ny * pars[4]
        a5 = pars[5]
        a6 = pars[6]
                        
        #The model is created using a Sersic profile             
        
        if (self.model_type == 'Sersic'):
            model_method = self.model_class.Sersic(amp_sersic=a0,r_eff_sersic=a1,\
                                   n_sersic=a2,x0_sersic=a3,\
                                       y0_sersic=a4,\
                                           ellip_sersic=a5,\
                                               theta_sersic=a6)                        
        #Convolution with the PSF
        modelin = self.conv2d_fft_psf(model_method)
                            
        logL_im = (self.yt - modelin)**2 / (self.noise_map)**2

        N = torch.sum(self.mask)

        chi2_unnormalized = torch.sum(logL_im * self.mask)
                    
        logL = -0.5 * chi2_unnormalized

        if (torch.isnan(logL)):
            logL = torch.inf
            print('hey')

        chi2 = chi2_unnormalized / N

        if numpy_input:            
            logL = logL.numpy()
            chi2 = chi2.numpy()
        
        return logL, chi2, modelin
    def hamiltonian_sersic(self):
        """
        Function to adjust the galaxy with the Sersic profile
        """
  

        # Because of the definition of the XY axes, theta=0 is horizontal and theta=90 is vertical
        self.left = torch.tensor([0.0, 0.0, 0.5, 0.1, 0.1, 0,   0])
        self.right = torch.tensor([6.0, 6.0, 10,  0.9, 0.9, 0.8, 180])
        pars = torch.zeros(7)
        pars = pars.clone().detach().requires_grad_(True)
        sampler = 'nested'
        self.rv = []
        for i in range(7):
            self.rv.append(stats.uniform(loc=self.left[i], scale=self.right[i]-self.left[i]))
        
        sampler = nestle.sample(self.logPosteriorSersic_logL, self.prior_transform_nested, 7, method='single',callback=nestle.print_progress)
        samples = nestle.resample_equal(sampler.samples, sampler.weights)
     
        samples[:, 0] = 10.0**samples[:, 0]
        samples[:, 1] = 10.0**samples[:, 1]
        samples[:, 3] = self.nx * samples[:, 3]
        samples[:, 4] = self.ny * samples[:, 4]
        return samples
