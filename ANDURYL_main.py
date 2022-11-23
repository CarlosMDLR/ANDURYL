# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:27:44 2022

@author: Usuario
"""

from astropy.io import fits
import matplotlib.pyplot as plt
from read_params import *
from priors import *
from profile_select import *
from profiles import *
from psf_generation import *
from hamiltonian_class import *
import cmasher as cmr
from astropy.convolution import convolve
# =============================================================================
# Eleccion de mapa de colores y lectura de datos
# =============================================================================
cmap = plt.get_cmap('cmr.redshift')
galaxies,Ie,Re,n, ba_b, PA_bulge, B_T, X_center, Y_center, chi_2= \
    data_reader(user_in_file)

data=fits.getdata(galaxies[0])

fig,ax = plt.subplots()
plt.imshow((data),cmap = cmap)

# =============================================================================
# Generacion y convolucion con la PSF
# =============================================================================
psf_class = psf(xsize,ysize,psf_imgname,gauss_amp,mean_x, mean_y, theta_rot,\
                stdv_x,stdv_y,moff_amp,moff_x, moff_y,width_moff,power_moff)
class_method = getattr(psf_class, psf_type)
psf_image = class_method() 
data_conv = convolve(data, psf_image)
fig,ax = plt.subplots()
plt.imshow((data_conv),cmap = cmap)

# =============================================================================
# Aplicacion del Hamiltonian
# =============================================================================
data_conv = data_conv.astype(np.float64)
hamiltonian_class = hamiltonian_model(data_conv)
class_method = getattr(hamiltonian_class, 'hamiltonian_sersic')
params = class_method() 