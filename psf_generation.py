# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 22:11:38 2022

@author: Carlos

A bayesiaN Decomposition code for Use in photometRY Labors
"""

from astropy.modeling.models import Gaussian2D,Moffat2D
import matplotlib.pyplot as plt
from read_params import *

def moff(x,y,fwhm,beta):
    xc,yc= int(len(x)/2),int(len(y)/2)
    rx=(x-xc)
    ry=(y-yc)
    r=np.sqrt(rx**2 + ry**2)
    alpha=fwhm/(2 *np.sqrt(2**(1/beta) -1))
    beta=beta
    moffat=((beta-1)/(alpha**2))*(1 +(r/alpha)**2)**(-beta)
    return(moffat)

class psf: 
    def __init__(self,xsize,ysize,psf_imgname,gauss_amp,mean_x, mean_y, theta_rot, stdv_x, \
        stdv_y,moff_amp,moff_x, moff_y,width_moff,power_moff):
        self.xsize = xsize
        self.ysize = ysize
        self.psf_imgname=psf_imgname
        self.gauss_amp=gauss_amp
        self.mean_x=mean_x
        self.mean_y=mean_y
        self.theta_rot=theta_rot
        self.stdv_x=stdv_x
        self.stdv_y=stdv_y
        self.moff_amp = moff_amp
        self.moff_x=moff_x
        self.moff_y=moff_y
        self.width_moff=width_moff
        self.power_moff=power_moff
    def Gaussian(self):
        y, x = np.mgrid[0:int(self.ysize), 0:int(self.xsize)]
        gm1 = Gaussian2D(self.gauss_amp,self.mean_x ,self.mean_y,self.stdv_x,self.stdv_y,self.theta_rot)
        g1 = gm1(x, y)
        g1 /= g1.sum()
        return(g1)
    def Moffat(self): 
        y, x = np.mgrid[0:int(self.ysize), 0:int(self.xsize)]
        m= moff(x,y,self.width_moff,self.power_moff)
        m /=m.sum()
        return(m)

    def Double_Gaussian(self):
        y, x = np.mgrid[0:int(self.ysize), 0:int(self.xsize)]
        gm1 = Gaussian2D(self.gauss_amp[0],self.mean_x[0] ,self.mean_y[0],self.stdv_x[0],self.stdv_y[0],self.theta_rot[0])
        gm2 = Gaussian2D(self.gauss_amp[1],self.mean_x[1] ,self.mean_y[1],self.stdv_x[1],self.stdv_y[1],self.theta_rot[1])
        g1 = gm1(x, y)
        g2 = gm2(x,y)
        g = g1+g2
        g /= g.sum()
        return(g)

