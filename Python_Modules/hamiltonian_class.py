# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:18:01 2022
@author: Usuario
"""

# =============================================================================
# Import modules
# =============================================================================
from priors import priors
from astropy.io import fits
from psf_generation import psf
from profile_select import profiles

import torch
import nestle_2
import importlib
import numpy as np
import read_params as paris
import scipy.special as sp
import scipy.stats as stats

# Torch is put to work in double precision and it is analyzed if simulations 
# or real galaxies are being used
torch.set_default_dtype(torch.float64)
use_simu = str(paris.use_simu).lower() in ['yes']
comm_params = str(paris.comm_params).lower() in ['yes']


def transform(param, a, b):
    """
    Return the transformed parameters using the logit transform to map 
    from (-inf,inf) to (a,b)
    It also returns the log determinant of the Jacobian of the transformation.
    """
    logit_1 = 1.0 / (1.0+ torch.exp(-param))
    transformed = a + (b-a) * logit_1
    logdet = torch.log(torch.tensor((b-a))) + torch.log(logit_1)\
        + torch.log((1.0 - logit_1))
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
    Class where the different adjustment methods are grouped 
    according to the different functional forms
    
    Implemented functions:
        -Sersic
        -Sersic+Exponential
        -Ferrers (in progress)
    """
    def __init__(self, image,width_moff,power_moff,mask,gain, noise_map,\
                 dir_img_final,name):

        # Here the mask is applied keeping the indices that have not 
        # been omitted
        if len(np.unique(mask))==1:
            self.mask = torch.tensor(1.0 - (mask-mask))
        else:
            self.mask = torch.tensor(1.0 - mask/np.max(mask))
        #---------------------------------------------------------------------#
        self.image = image #image of the galaxy to adjust
        self.psf = psf #psf previously generated in the main program 
        self.gain=gain
        self.name=name
        self.dir_img_final=dir_img_final
        self.noise_map = torch.tensor(noise_map)
        self.model_class = \
            profiles(y_size=self.image.shape[0],x_size=self.image.shape[1])


        self.witdh_init = width_moff #psf parameters for priors limits
        self.power_init=power_moff
        self.yt = torch.tensor(self.image)

        self.ny, self.nx = self.yt.shape # shape of the galaxy image

    def conv2d_fft_psf(self, f,psf_fft):
        """
        Function that performs the convolution with the psf
        """
        ff = torch.fft.fft2(f)
        return (torch.fft.ifft2(ff * psf_fft)).real
    
    # Specialized functions to return the logL and chi2 only
    def logPosteriorSersic_logL(self, pars):
        with torch.no_grad():
            logL, chi2, model = self.logPosteriorSersic(pars,numpy_input=True)
        return logL
    def logPosteriorSersicExp_logL(self, pars):
        with torch.no_grad():
            logL,chi2,model=self.logPosteriorSersicExp(pars, numpy_input=True)
        return logL   
    def logPosteriorSersic_chi2(self, pars):
        logL, chi2, model = self.logPosteriorSersic(pars)
        return chi2
    def logPosteriorSersicExp_chi2(self, pars):
        logL, chi2, model = self.logPosteriorSersicExp(pars)
        return chi2    
       
    def prior_transform_nested(self, u):
        #transformation of priors for nested sampling
        x = np.array(u)
        for i, rv in enumerate(self.rv):
            x[i] = rv.ppf(u[i])
                
        return x

    def logPosteriorSersic(self, pars, numpy_input=False):
        """
        This function calculates the logL for a fit with a Sersic profile.
        
        List of params:
          -pars[0]= Ie
          -pars[1]= Re
          -pars[2]= n
          -pars[3]= x0
          -pars[4]= y0
          -pars[5]= ellip
          -pars[6]= theta 
          -pars[7]= fwhm
          -pars[8]= beta
          -pars[9]= noise
        """

        pars = torch.tensor(pars)

        a0 = 10.0**pars[0]
        a1 = 10.0**pars[1]
        a2 = pars[2]
        a3 = self.nx * pars[3]
        a4 = self.ny * pars[4]
        a5 = pars[5]
        a6 = pars[6]
        a7 = pars[7]   
        a8 = pars[8]
        a9 = pars[9]
        
        #The model is created using a Sersic profile             

        model_method = self.model_class.Sersic(amp_sersic=a0,r_eff_sersic=a1,\
                               n_sersic=a2,x0_sersic=a3,\
                                   y0_sersic=a4,\
                                       ellip_sersic=a5,\
                                           theta_sersic=a6) 
        # =====================================================================
        # Generation of the PSF
        # ===================================================================== 
        psf_class = psf(paris.xsize,paris.ysize,paris.gauss_amp,paris.mean_x,\
                        paris.mean_y, paris.theta_rot,paris.stdv_x,\
                        paris.stdv_y,paris.moff_amp,paris.xsize//2,\
                        paris.ysize//2,a7,a8,None, None, None,\
                        None,None,None,None)
        class_method = getattr(psf_class, paris.psf_type)
        psf_image = class_method() 
        # Here we proceed to carry out previous steps for the convolution with 
        # the psf. You have to do zero padding and be careful where python 
        # starts to perform the convolution. That is why they are made 
        # iffftshift now
        sz = (self.image.shape[0] - psf_image.shape[0],\
              self.image.shape[1] - psf_image.shape[1])
        kernel = np.pad(psf_image, (((sz[0]+1)//2, sz[0]//2),\
                                    ((sz[1]+1)//2,sz[1]//2)))
        psf_mod = torch.tensor(kernel)
        psf_fft = torch.fft.fft2( torch.fft.ifftshift(psf_mod))                     
        # Convolution with the PSF
        modelin = self.conv2d_fft_psf(model_method,psf_fft)+a9        
         
        # Calculation of logL and chi^2
        logL_im = (self.yt - modelin)**2 / (self.noise_map)**2

        N = torch.sum(self.mask)

        chi2_unnormalized = torch.sum(logL_im * self.mask)
              
        logL = -0.5 * chi2_unnormalized

        if (torch.isnan(logL)):
            # Just a warning in case infinities are detected in the logL
            logL = torch.tensor(torch.inf)
            print('\n Ahhhhhh infinite in logL')

        chi2 = chi2_unnormalized / N

                        
        logL = logL.numpy()
        chi2 = chi2.numpy()
        
        return logL, chi2, modelin
    
    
    def logPosteriorSersicExp(self, pars, numpy_input=False):
        """
        This function calculates the logL for a fit with a Sersic+exp profile.
        
        List of params:
          -pars[0]= Ie
          -pars[1]= Re
          -pars[2]= n
          -pars[3]= x0
          -pars[4]= y0
          -pars[5]= ellip
          -pars[6]= theta 
          -pars[7]= fwhm
          -pars[8]= beta
          -pars[9]= noise
          -pars[10]= I0
          -pars[11]= Re/h
          -pars[12]= ellip_exp
          -pars[13]= PA_exp
        """           
        
        
        pars = torch.tensor(pars)
       
        a0 = 10.0**pars[0]
        a1 = 10.0**pars[1]
        a2 = pars[2]
        a3 = self.nx * pars[3]
        a4 = self.ny * pars[4]
        a5 = pars[5]
        a6 = pars[6] 
        a7 = pars[7]   
        a8 = pars[8]
        a9 = pars[9]
        a10 = 10.0**pars[10]
        a11 = pars[11]
        a12 = pars[12] 
        a13 = pars[13]
        
        #The model is created using a Sersic+exp profile             
        model_method_sersic = self.model_class.Sersic(amp_sersic=a0,\
                               r_eff_sersic=a1,n_sersic=a2,x0_sersic=a3,\
                               y0_sersic=a4, ellip_sersic=a5,theta_sersic=a6)  
        h = a1/a11
        model_method_exp = self.model_class.Exponential(amp_expo=a10,\
                               r_eff_expo=h, x0_expo=a3,y0_expo=a4,\
                               ellip_expo=a12, theta_expo=a13)
        model_method=model_method_sersic+model_method_exp

        # =====================================================================
        # Generation of the PSF
        # ===================================================================== 
        psf_class = psf(paris.xsize,paris.ysize,paris.gauss_amp,paris.mean_x,\
                        paris.mean_y, paris.theta_rot,paris.stdv_x,\
                        paris.stdv_y,paris.moff_amp,paris.xsize//2,\
                        paris.ysize//2,a7,a8,None, None, None, None,\
                        None,None,None)
        class_method = getattr(psf_class, paris.psf_type)
        psf_image = class_method() 
        # Here we proceed to carry out previous steps for the convolution with 
        # the psf. You have to do zero padding and be careful where python 
        # starts to perform the convolution. That is why they are made 
        # iffftshift now
        sz = (self.image.shape[0] - psf_image.shape[0],\
              self.image.shape[1] - psf_image.shape[1])
        kernel = np.pad(psf_image, (((sz[0]+1)//2, sz[0]//2),\
                                    ((sz[1]+1)//2,sz[1]//2)))
        psf_mod = torch.tensor(kernel)
        psf_fft = torch.fft.fft2( torch.fft.ifftshift(psf_mod))                     
        #Convolution with the PSF
        
    
        modelin = self.conv2d_fft_psf(model_method,psf_fft)+a9
        
        # Calculation of logL and chi^2                    
        logL_im = (self.yt - modelin)**2 / (self.noise_map)**2

        N = torch.sum(self.mask)
        
        chi2_unnormalized = torch.sum(logL_im * self.mask)

        logL = -0.5 * chi2_unnormalized

        if (torch.isnan(logL)):
            # Just a warning in case infinities are detected in the logL            
            logL = torch.tensor(torch.inf)
            print('\n Ahhhhhh infinite in logL')

        chi2 = chi2_unnormalized / N                
        logL = logL.numpy()
        chi2 = chi2.numpy()
        
        return logL, chi2, modelin
    
    def hamiltonian_sampling(self):
        """
        Function to perform Bayesian sampling
        """
  
        if paris.to_fit=="Bulge-disc":
            # Perform the fit for a Sersic+exp profile   
            
            if (self.power_init-10)<=0:lim_inf_pow=0.1
            else:lim_inf_pow=self.power_init-10
            
            # The lower and higher limits of the priors of the different
            # parameters are established
            self.left = torch.tensor([ 0.0, 0.0, 0.5, 0.1, 0.1, 0,\
                                      0, 1.0,lim_inf_pow,0, 0.0, 0.05,0,0])
            self.right = torch.tensor([6.0, 6.0, 10,  0.9, 0.9, 0.8, 180,5,\
                                       18,paris.sky_sigm,6.0, 1.678,0.8, 180])
    
            pars = torch.zeros(14)
            pars = pars.clone().detach().requires_grad_(True)
            
            self.rv = []
            # The transformations of the priors are performed
            for i in range(14):
                self.rv.append(stats.uniform(loc=self.left[i],\
                                             scale=self.right[i]-self.left[i]))
            
            # Sampling is perform
            try:
                importlib.reload(nestle_2)#for simulations enlarge=0.3,npoints=110
               
                sampler = nestle_2.sample(self.logPosteriorSersicExp_logL,\
                                          self.prior_transform_nested,14,\
                                          method='multi',enlarge=0.3,\
                                          npoints=110,\
                                          callback=nestle_2.print_progress)
                samples = nestle_2.resample_equal(sampler.samples,\
                                                  sampler.weights)
            # If the sampling takes more than 80 minutes to complete,
            # proceed to the next photometric adjustment
            except TimeoutError:
                print("\n Exception at gal %s"%(self.name[:-5]))
                print("\n The program has taken more than 80 minutes to run, it proceeds to the next galaxy")
                importlib.reload(nestle_2)
                samples=None
            # If there is any other type of error, proceed to 
            # the next photometric adjustment    
            except:
                print("\n Exception at gal %s"%(self.name[:-5]))
                importlib.reload(nestle_2)
                samples=None
            # Samples are saved for each parameter
            try:
                samples[:, 0] = 10.0**samples[:, 0]
                samples[:, 1] = 10.0**samples[:, 1]
                samples[:, 3] = self.nx * samples[:, 3]
                samples[:, 4] = self.ny * samples[:, 4]
                samples[:, 10] = 10.0**samples[:, 10]
                samples[:, 11] = samples[:, 1]/samples[:, 11]
                if use_simu:
                    hdu = fits.PrimaryHDU(samples)
                    hdu.writeto(self.dir_img_final\
                                +"/simulation_samples_bulge_disc"\
                                +"/sample_bulge_disc"+ self.name[:-5]\
                                +'.fits',overwrite=True)
                if not use_simu:
                    hdu = fits.PrimaryHDU(samples)
                    hdu.writeto(self.dir_img_final+"/gals_samples_bulge_disc"\
                                +"/sample_bulge_disc"+ self.name[:-5]\
                                +'.fits',overwrite=True)
            except:
                samples=None
            return samples
        
        elif paris.to_fit=="Bulge":
            # Perform the fit for a Sersic profile
            if (self.power_init-10)<=0:lim_inf_pow=0.1
            else:lim_inf_pow=self.power_init-10
            # The lower and higher limits of the priors of the different
            # parameters are established
            self.left = torch.tensor([ 0.0, 0.0, 0.5, 0.1, 0.1, 0,   0,\
                                      1.0,lim_inf_pow,0])
            self.right = torch.tensor([6.0, 6.0, 10,  0.9, 0.9, 0.8, 180,\
                                       5,18,paris.sky_sigm])

            pars = torch.zeros(10)
            pars = pars.clone().detach().requires_grad_(True)
 
            self.rv = []
            # The transformations of the priors are performed
            for i in range(10):
                self.rv.append(stats.uniform(loc=self.left[i],\
                                             scale=self.right[i]-self.left[i]))
            # Sampling is perform
            try:
                importlib.reload(nestle_2)#for simulations enlarge=0.3,npoints=110
               
                sampler = nestle_2.sample(self.logPosteriorSersic_logL,\
                                          self.prior_transform_nested,10,\
                                          method='multi',enlarge=0.6,\
                                          callback=nestle_2.print_progress)
                samples = nestle_2.resample_equal(sampler.samples,\
                                                  sampler.weights)
            # If the sampling takes more than 80 minutes to complete,
            # proceed to the next photometric adjustment
            except TimeoutError:
                print("\n Exception at gal %s"%(self.name[:-5]))
                print("\n The program has taken more than 80 minutes to run, it proceeds to the next galaxy")
                importlib.reload(nestle_2)
                samples=None
            # If there is any other type of error, proceed to 
            # the next photometric adjustment    
            except:
                print("\n Exception at gal %s"%(self.name[:-5]))
                importlib.reload(nestle_2)
                samples=None
            # Samples are saved for each parameter    
            try:
                samples[:, 0] = 10.0**samples[:, 0]
                samples[:, 1] = 10.0**samples[:, 1]
                samples[:, 3] = self.nx * samples[:, 3]
                samples[:, 4] = self.ny * samples[:, 4]
                if use_simu:
                    hdu = fits.PrimaryHDU(samples)
                    hdu.writeto(self.dir_img_final\
                                +"/simulation_samples_bulge"\
                                +"/sample_bulge"+ self.name[:-5]\
                                +'.fits',overwrite=True)
                if not use_simu:
                    hdu = fits.PrimaryHDU(samples)
                    hdu.writeto(self.dir_img_final+"/gals_samples_bulge"\
                                +"/sample_bulge"+ self.name[:-5]+'.fits',\
                                overwrite=True)
            except:
                print("\n Exception at gal %s"%(self.name[:-5]))
                samples =None
            return samples
