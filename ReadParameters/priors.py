# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:08:50 2022

@author: Usuario
"""

import numpy as np
from astropy.modeling.models import Sersic2D
from profiles import Exponential2D,Ferrers2D

class priors(self,):
    
    def Uprior(self):
        return(np.random.uniform(A,B,(N,N)))
    def Gprior(self):
        return(np.random.normal(loc=means,scale =sigmas))
            