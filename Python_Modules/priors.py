# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:08:50 2022

@author: Usuario
"""

import numpy as np

class priors:
    def __init__(self,A,B,N):
        self.A =A
        self.B=B
        self.N =N
    def Uprior(self):
        return(np.random.uniform(self.A,self.B,(self.N,self.N)))

            