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
from profiles_torch import *
from psf_generation import *
from hamiltonian_class import *
import cmasher as cmr
from astropy.convolution import convolve

# =============================================================================
# Choosing color map and reading data
# =============================================================================
cmap = plt.get_cmap('cmr.redshift')
galaxies,Ie,Re,n, ba_b, PA_bulge, B_T, X_center, Y_center, chi_2= \
    data_reader(user_in_file)

data=fits.getdata(galaxies[0])

box = data[int(Y_center)-10:int(Y_center )+10,int(X_center)-10:int(X_center)+10]
norm_I = np.max(box)
data = data/norm_I
fig,ax = plt.subplots()
mapi=plt.imshow((data*norm_I),cmap = cmap)
plt.colorbar(mapi)
plt.title("Datos")
# =============================================================================
# Generation of the PSF
# =============================================================================
nx, ny = data.shape
moff_x = nx / 2.0
moff_y = ny / 2.0

psf_class = psf(nx,ny,psf_imgname,gauss_amp,mean_x, mean_y, theta_rot,\
                stdv_x,stdv_y,moff_amp,moff_x, moff_y,width_moff,power_moff)
class_method = getattr(psf_class, psf_type)
psf_image = class_method() 
fig,ax = plt.subplots()
mapi=plt.imshow((psf_image),cmap = cmap)
plt.colorbar(mapi)
plt.title("PSF_Moffat")
# =============================================================================
# Application of the Hamiltorch
# =============================================================================

hamiltonian_class = hamiltonian_model(data.astype(np.float64),psf_image)
class_method = getattr(hamiltonian_class, 'hamiltonian_sersic')
params = class_method() 

# =============================================================================
#  Model print
# =============================================================================
params2=torch.mean(params,axis=(0))
model_class = profiles(x_size=data.shape[0],y_size=data.shape[1])
model_method = model_class.Sersic(amp_sersic=params2[0],\
                                  r_eff_sersic=params2[1], n_sersic=params2[2]\
                                      ,x0_sersic=params2[3], \
                                          y0_sersic=params2[4],\
                                              ellip_sersic=params2[5], \
                                                  theta_sersic=params2[6])
ma = model_method
a = ma.detach().numpy()
b =  convolve(a, psf_image)
fig,ax = plt.subplots()
mapi=plt.imshow( b*norm_I,cmap = cmap)
plt.colorbar(mapi)
plt.title("Modelo")
