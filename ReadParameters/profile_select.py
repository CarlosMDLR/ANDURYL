# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:35:09 2022

@author: Usuario
"""
import numpy as np
from profiles import Sersic2D,Exponential2D,Ferrers2D
import matplotlib.pyplot as plt
class profiles:
    def __init__(self,x_size=100,y_size=100,amp_sersic=1, r_eff_sersic=25, n_sersic=1, x0_sersic=50,y0_sersic=50, ellip_sersic=.5, theta_sersic=20, \
                amp_exp=1,h_exp=20,x0_exp=50,y0_exp=50,ellip_exp=.5,theta_exp=20,\
                    amp_ferrers=1, a_bar_ferrers=5, n_bar_ferrers=3, x0_ferrers=50,y0_ferrers=50, ellip_ferrers=.5, theta_ferrers=20):
        self.x_size = x_size
        self.y_size = y_size
        #Sersic params
        self.amp_sersic = amp_sersic
        self.r_eff_sersic= r_eff_sersic
        self.n_sersic=n_sersic
        self.x0_sersic=x0_sersic
        self.y0_sersic=y0_sersic
        self.ellip_sersic=ellip_sersic
        self.theta_sersic=theta_sersic
        #Exp params
        self.amp_exp = amp_exp
        self.h_exp= h_exp
        self.x0_exp=x0_exp
        self.y0_exp=y0_exp
        self.ellip_exp=ellip_exp
        self.theta_exp=theta_exp  
        #Ferrers params
        self.amp_ferrers = amp_ferrers
        self.a_bar_ferrers= a_bar_ferrers
        self.n_bar_ferrers=n_bar_ferrers
        self.x0_ferrers=x0_ferrers
        self.y0_ferrers=y0_ferrers
        self.ellip_ferrers=ellip_ferrers
        self.theta_ferrers=theta_ferrers
    def Sersic(self):
        x,y = np.meshgrid(np.arange(int(self.x_size)), np.arange(int(self.y_size)))
        model = Sersic2D(amplitude = self.amp_sersic, r_eff =self.r_eff_sersic, n= self.n_sersic, x_0= self.x0_sersic, y_0= self.y0_sersic,
               ellip=self.ellip_sersic, theta=self.theta_sersic)
        image = model(x,y)
        return(image)
    def Exponential(self):
        x,y = np.meshgrid(np.arange(int(self.x_size)), np.arange(int(self.y_size)))
        model = Exponential2D(amplitude = self.amp_exp, h =self.h_exp, x_0= self.x0_exp, y_0= self.y0_exp,
               ellip=self.ellip_exp, theta=self.theta_exp)
        image = model(x,y)
        return(image)
    def Ferrers(self):
        x,y = np.meshgrid(np.arange(int(self.x_size)), np.arange(int(self.y_size)))
        model = Ferrers2D(amplitude = self.amp_ferrers,a_bar =self.a_bar_ferrers,n_bar=self.n_bar_ferrers , x_0= self.x0_ferrers, y_0= self.y0_ferrers,
               ellip=self.ellip_ferrers, theta=self.theta_ferrers)
        image = model(x,y)
        return(image)

"""
clas = profiles()

ser = clas.Sersic()
exp = clas.Exponential()
fer = clas.Ferrers()

plt.figure()
plt.imshow((ser))
plt.figure()
plt.imshow((exp))
plt.figure()
plt.imshow((fer))
"""