# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:35:09 2022

@author: Usuario
"""
# Modules are imported
import torch
from profiles_torch import Sersic2D,Exponential2D,Ferrers2D

# Torch is put to work in double precision
torch.set_default_dtype(torch.float64)
class profiles:
    """ 
    This class creates the 2D mesh, according to the object to be modeled,
    and paints the generated model (Sersic or exponential) on it.
    """
    def __init__(self,x_size=100,y_size=100,rebin_size=3):
        self.x_size = x_size
        self.y_size = y_size
        self.rebin_size=rebin_size
        
        self.y, self.x = torch.meshgrid(torch.arange(int(self.y_size)),\
                                        torch.arange(int(self.x_size)))

    def Sersic(self, amp_sersic=1, r_eff_sersic=25, n_sersic=1, x0_sersic=50,\
               y0_sersic=50, ellip_sersic=.5, theta_sersic=20):            
        model = Sersic2D(x=self.x,y=self.y,amplitude = amp_sersic,\
                         r_eff =r_eff_sersic, n= n_sersic, x_0= x0_sersic,\
                         y_0= y0_sersic,ellip=ellip_sersic, theta=theta_sersic)
        return(model())
    def Exponential(self, amp_expo=1, r_eff_expo=25, x0_expo=50,y0_expo=50,\
                    ellip_expo=.5, theta_expo=20):
        model = Exponential2D(x=self.x,y=self.y,amplitude = amp_expo,\
                              r_eff =r_eff_expo, x_0= x0_expo, y_0= y0_expo,\
                              ellip=ellip_expo, theta=theta_expo)
        return(model())
# =============================================================================
#     IN PREPARATION
# =============================================================================
    def Ferrers(self):
        x,y = torch.meshgrid(torch.arange(int(self.x_size)), torch.arange(int(self.y_size)))
        model = Ferrers2D(x=x,y=y,amplitude = self.amp_ferrers,a_bar =self.a_bar_ferrers,n_bar=self.n_bar_ferrers , x_0= self.x0_ferrers, y_0= self.y0_ferrers,
                ellip=self.ellip_ferrers, theta=self.theta_ferrers)
        return(model())
