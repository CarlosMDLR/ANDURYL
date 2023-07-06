# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 22:11:38 2022

@author: Carlos

A bayesiaN Decomposition code for Use in photometRY Labors
"""
# Modules are imported
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Gaussian2D


def moff(x,y,fwhm,beta):
    """ 
    This function generates a Moffatian PSF on a 2D mesh.
    """
    xc,yc= int(len(x)/2),int(len(y)/2)
    rx=(x-xc)
    ry=(y-yc)
    r=np.sqrt(rx**2 + ry**2)
    alpha=fwhm/(2 *np.sqrt(2**(1/beta) -1))
    beta=beta
    moffat=((beta-1)/(alpha**2))*(1 +(r/alpha)**2)**(-beta)
    return(moffat)

def reconstructPSF(psFieldFilename, filterin, row, col):
    """ 
    This function rebuilds the SDSS PSF from its data, 
    but is not currently used in the code.
    """
    filterIdx = 'ugriz'.index(filterin) + 1
    psField = fits.open(psFieldFilename)
    
    pStruct = psField[filterIdx].data

    nrow_b = pStruct['nrow_b'][0]
    ncol_b = pStruct['ncol_b'][0]

    rnrow = pStruct['rnrow'][0]
    rncol = pStruct['rncol'][0]

    nb = nrow_b * ncol_b
    coeffs = np.zeros(nb.size, float)
    ecoeff = np.zeros(3, float)
    cmat = pStruct['c']

    rcs = 0.001
    for ii in range(0, nb.size):
        coeffs[ii] = (row * rcs)**(ii % nrow_b) * (col * rcs)**(ii / nrow_b)

    for jj in range(0, 3):
        for ii in range(0, nb.size):
            ecoeff[jj] = ecoeff[jj] + cmat[int(ii / nrow_b),\
                                           ii % nrow_b, jj] * coeffs[ii]

    psf = pStruct['rrows'][0] * ecoeff[0] + \
        pStruct['rrows'][1] * ecoeff[1] + \
        pStruct['rrows'][2] * ecoeff[2]

    psf = np.reshape(psf, (rnrow, rncol))
    # psf = psf[10:40, 10:40]  # Trim non-zero regions.
    psf /=psf.sum()
    return psf

def power_law_sdss(x,y,photField,gal_ID,ind):
    """ 
    This function rebuilds the SDSS power law PSF from its data, 
    but is not currently used in the code.
    """
    xc,yc= int(len(x)/2),int(len(y)/2)
    rx=(x-xc)
    ry=(y-yc)
    r=np.sqrt(rx**2 + ry**2)
    gal_IDs = photField["field"]
    
    ind_filt = np.where((gal_ID ==gal_IDs))[0][0]
    b = photField["psf_b"][ind_filt][ind]
    beta = photField["psf_beta"][ind_filt][ind]
    sigma1=photField["psf_sigma1"][ind_filt][ind]
    sigma2=photField["psf_sigma2"][ind_filt][ind]
    sigmap=photField["psf_sigmap"][ind_filt][ind]
    p0=photField["psf_p0"][ind_filt][ind]
    
    power=np.exp(-0.5*(r/sigma1)**2) + b*np.exp(-0.5*(r/sigma2)**2)\
        + p0*((1.+(((r/sigmap)**2)/beta))**(- beta / 2))
    return(power)
class psf: 
    """ 
    This class paints the PSF model in a 2D mesh, depending on the type 
    of PSF you want to generate, currently only the Moffat is in use.
    """
    def __init__(self,xsize,ysize,gauss_amp,mean_x, mean_y, theta_rot, stdv_x,\
        stdv_y,moff_amp,moff_x, moff_y,width_moff,power_moff,psFieldFilename,\
            filterin, row, col,photField,gal_ID,ind_filt):
        self.xsize = xsize
        self.ysize = ysize
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
        self.psFieldFilename=psFieldFilename
        self.filter=filterin
        self.row=row
        self.col=col
        self.photfield = photField
        self.gal_ID = gal_ID
        self.ind_filt=ind_filt
    def Gaussian(self):
        x,y = np.mgrid[0:int(self.ysize), 0:int(self.xsize)]
        gm1 = Gaussian2D(self.gauss_amp,self.mean_x ,self.mean_y,\
                         self.stdv_x,self.stdv_y,self.theta_rot)
        g1 = gm1(x, y)
        g1 /= g1.sum()
        return(g1)
    def Moffat(self): 
        x,y = np.mgrid[0:int(self.ysize), 0:int(self.xsize)]
        m= moff(x,y,self.width_moff,self.power_moff)
        m /=m.sum()
        return(m)
    def SDSS(self):
        psf=reconstructPSF(self.psFieldFilename,self.filter,self.row,self.col)
        return psf
    def SDSS_power_law(self):
        x,y = np.mgrid[0:int(self.ysize), 0:int(self.xsize)]
        power = power_law_sdss(x,y,self.photfield,self.gal_ID,self.ind_filt)
        power /=power.sum()
        return(power)
    def Double_Gaussian(self):
        x,y = np.mgrid[0:int(self.ysize), 0:int(self.xsize)]
        gm1 = Gaussian2D(self.gauss_amp[0],self.mean_x[0] ,self.mean_y[0],\
                         self.stdv_x[0],self.stdv_y[0],self.theta_rot[0])
        gm2 = Gaussian2D(self.gauss_amp[1],self.mean_x[1] ,self.mean_y[1],\
                         self.stdv_x[1],self.stdv_y[1],self.theta_rot[1])
        g1 = gm1(x, y)
        g2 = gm2(x,y)
        g = g1+g2
        g /= g.sum()
        return(g)

