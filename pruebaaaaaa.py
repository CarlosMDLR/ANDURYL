# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 19:03:56 2023

@author: Cintia_Marrero
"""
from astropy.io import fits
import matplotlib.pyplot as plt
from read_params import *
from priors import *
from profile_select import *
from profiles_torch import *
from psf_generation import *
from hamiltonian_class import *
import cmasher as cmr
from astropy.convolution import convolve
from matplotlib.ticker import FormatStrFormatter
from getdist import plots, gaussian_mixtures,MCSamples
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from time import time
cmap = plt.get_cmap('cmr.torch')
#data = fits.getdata("./Simulacion_Jairo/image_noconv.fits")
data = fits.getdata("model_simulation_sin_over.fits")
paramis = np.array([ 45.715599,20.487037,5.0000000,150.38,150.38,1- 0.77361612,156.73182])
params2=torch.tensor(paramis)
model_class = profiles(x_size=data.shape[1],y_size=data.shape[0])
model_method = model_class.Sersic(amp_sersic=params2[0],\
                                  r_eff_sersic=params2[1], n_sersic=params2[2]\
                                      ,x0_sersic=params2[3], \
                                          y0_sersic=params2[4],\
                                              ellip_sersic=params2[5], \
                                                  theta_sersic=params2[6])
ma = model_method
b =  ma

residual = (b.detach().numpy())-data
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
mapi =ax[0].imshow((data),cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[0],shrink=0.5,extend='both')
cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[0].set_title("Data. Grid size: 11")
mapi = ax[1].imshow( b.detach().numpy(),cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[1],shrink=0.5,extend='both')
cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[1].set_title("Model. Central pix val: %.3f"%(b.detach().numpy()[150,150]))
mapi = ax[2].imshow(residual[135:165,135:165],cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[2],shrink=0.5,extend='both')
cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[2].set_title("Residual map")
plt.tick_params(axis="x", direction="in", length=7, width=1.2, color="k")
plt.tick_params(axis="y", direction="in", length=7, width=1.2, color="k")
ax[0].set_ylabel(r'y [px]', fontsize = 16)
ax[0].set_xlabel(r'x [px]', fontsize =16)
ax[0].xaxis.set_minor_locator(AutoMinorLocator())
ax[0].yaxis.set_minor_locator(AutoMinorLocator())
ax[0].tick_params(direction="in",which='minor', length=4, color='k')
ax[1].set_ylabel(r'y [px]', fontsize = 16)
ax[1].set_xlabel(r'x [px]', fontsize =16)
ax[1].xaxis.set_minor_locator(AutoMinorLocator())
ax[1].yaxis.set_minor_locator(AutoMinorLocator())
ax[1].tick_params(direction="in",which='minor', length=4, color='k')
ax[2].set_ylabel(r'y [px]', fontsize = 16)
ax[2].set_xlabel(r'x [px]', fontsize =16)
ax[2].xaxis.set_minor_locator(AutoMinorLocator())
ax[2].yaxis.set_minor_locator(AutoMinorLocator())
ax[2].tick_params(direction="in",which='minor', length=4, color='k')
plt.show()