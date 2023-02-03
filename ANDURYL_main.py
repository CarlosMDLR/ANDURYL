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
from matplotlib.ticker import FormatStrFormatter
# =============================================================================
# Choosing color map and reading data
# =============================================================================
cmap = plt.get_cmap('cmr.redshift')
galaxies,Ie,Re,n, ba_b, PA_bulge, B_T, X_center, Y_center, chi_2= \
    data_reader(user_in_file)

data=fits.getdata(galaxies[0])*gain
mask = fits.getdata("I_recor_mask.fits")
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
ny, nx = data.shape
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
#npix=nx*ny
#noise = np.sqrt(sky_sigm*npix+(readout_noise**2)*npix)/norm_I
hamiltonian_class = hamiltonian_model(data.astype(np.float64),psf_image,mask,sky_sigm,readout_noise,gain,norm_I)
class_method = getattr(hamiltonian_class, 'hamiltonian_sersic')
params = class_method() 
comprobar = params.detach().numpy()
# =============================================================================
#  Model print
# =============================================================================
params2=torch.mean(params,axis=(0))
model_class = profiles(x_size=data.shape[1],y_size=data.shape[0])
model_method = model_class.Sersic(amp_sersic=params2[0],\
                                  r_eff_sersic=params2[1], n_sersic=params2[2]\
                                      ,x0_sersic=params2[3], \
                                          y0_sersic=params2[4],\
                                              ellip_sersic=params2[5], \
                                                  theta_sersic=params2[6])
ma = model_method
b =  hamiltonian_class.conv2d_fft_psf(ma)
fig,ax = plt.subplots()
mapi=plt.imshow( b.detach().numpy()*norm_I,cmap = cmap)
plt.colorbar(mapi)
plt.title("Modelo")

fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(10, 15))
ax[0][0].hist(comprobar[:,0],label=r"$I_e$");ax[0][0].legend();ax[0][0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1][0].hist(comprobar[:,1],label=r"$R_e$");ax[1][0].legend();ax[1][0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[2][0].hist(comprobar[:,2],label=r"$n$");ax[2][0].legend();ax[2][0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[3][0].hist(comprobar[:,3],label=r"$x_0$");ax[3][0].legend();ax[3][0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[0][1].hist(comprobar[:,4],label=r"$y_0$");ax[0][1].legend();ax[0][1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1][1].hist(comprobar[:,5],label=r"$\varepsilon$");ax[1][1].legend();ax[1][1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[2][1].hist(comprobar[:,6],label=r"$\theta$");ax[2][1].legend();ax[2][1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[3][1].imshow( b.detach().numpy()*norm_I,cmap = cmap)