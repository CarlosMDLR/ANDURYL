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
from getdist import plots, gaussian_mixtures,MCSamples
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from time import time

# Start counting
start_time = time()
# =============================================================================
# Choosing color map and reading data
# =============================================================================
ruta = './for_TEST/'
cmap = plt.get_cmap('cmr.redshift')
galaxies,Ie,Re,n, ba_b, PA_bulge, B_T, X_center, Y_center, chi_2= \
    data_reader(user_in_file)

data=fits.getdata(ruta+galaxies[0])*gain
data = data[:,:]
mask = fits.getdata(ruta+"I_recor_mask.fits")
mask=mask[:,:]
box = data[int(Y_center)-10:int(Y_center )+10,int(X_center)-10:int(X_center)+10]
norm_I = np.max(box)
data = data/norm_I

# =============================================================================
# Generation of the PSF
# =============================================================================
ny, nx = data.shape
moff_x = np.floor(nx / 2.0)
moff_y = np.floor(ny / 2.0)

# psf_class = psf(nx,ny,psf_imgname,gauss_amp,mean_x, mean_y, theta_rot,\
#                 stdv_x,stdv_y,moff_amp,moff_x, moff_y,width_moff,power_moff)
psf_class = psf(21,21,psf_imgname,gauss_amp,mean_x, mean_y, theta_rot,\
                stdv_x,stdv_y,moff_amp,10, 10,width_moff,power_moff)
class_method = getattr(psf_class, psf_type)
psf_image = class_method() 
# fig,ax = plt.subplots()
# mapi=plt.imshow((psf_image),cmap = cmap)
# plt.colorbar(mapi)
# plt.title("PSF_Moffat")
# =============================================================================
# Application of the Hamiltorch
# =============================================================================

hamiltonian_class = hamiltonian_model(data.astype(np.float64),psf_image,mask,sky_sigm,readout_noise,gain,norm_I)
class_method = getattr(hamiltonian_class, 'hamiltonian_sersic')
params,burn,step_size,L,N = class_method() 
comprobar = params.detach().numpy()
# =============================================================================
#  Model calculation
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
# =============================================================================
# Residual calculation and data-model-residual plot
# =============================================================================

residual = b.detach().numpy()-data
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
mapi =ax[0].imshow((data*norm_I),cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[0],shrink=0.5,extend='both')
cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[0].set_title("Data")
mapi = ax[1].imshow( b.detach().numpy()*norm_I,cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[1],shrink=0.5,extend='both')
cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[1].set_title("Model")
mapi = ax[2].imshow(residual*norm_I,vmin=-5000,vmax=3000,cmap = cmap)
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

#In logarithm

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
mapi =ax[0].imshow(np.log10(data*norm_I),cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[0],shrink=0.5,extend='both')
cbar.set_label(r"log I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[0].set_title("Data")
mapi = ax[1].imshow(np.log10(b.detach().numpy()*norm_I),cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[1],shrink=0.5,extend='both')
cbar.set_label(r"log I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[1].set_title("Model")
mapi = ax[2].imshow((abs(residual)/residual)*np.log10(abs(residual*norm_I)),cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[2],shrink=0.5,extend='both')
cbar.set_label(r"log I [$e^{-}$]",loc = 'center',fontsize = 16)
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
# =============================================================================
#  Triangular plot
# =============================================================================
g = plots.get_subplot_plotter(width_inch=20)
g.settings.lab_fontsize = 14
g.settings.axes_fontsize = 14
g.settings.axis_tick_x_rotation=45
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
g.settings.legend_fontsize=16
g.settings.axis_tick_max_labels=3
g.settings.title_limit_fontsize = 13
g.settings.prob_y_ticks=True
samples = MCSamples(samples=comprobar,names=['I0','Re','n','x0','y0','varepsilon','PA'],labels=['I_0^{norm}', 'R_e [px]','n','x_0 [px]','y_0 [px]',r'\varepsilon','PA [º]'])
g.triangle_plot([samples],
    filled=True,
    legend_labels=[r'$I_0$=0.3, $R_e$=100, n=0.5, $x_0$=120, $y_0$=110, $\varepsilon$=0.4, PA=30'], 
    line_args=[{'ls':'--', 'color':'green'},
               {'lw':2, 'color':'darkblue'}], 
    contour_colors=['darkblue'],
    title_limit=1, # first title limit (for 1D plots) is 68% by default
    markers={'x2':0})
plt.suptitle('Burn=%.0f ; Step=%.2e ; L=%.0f ; N=%.0f'%(burn,step_size,L,N), va='bottom',fontsize=14)
#plt.savefig('\Documentos\Master Astrofisica\TFM\ANDURYL\Figures'+ '\%s'%("figura_prueba2_sinover"), bbox_inches='tight', pad_inches=0.02)

# Calculate the elapsed time
elapsed_time = time() - start_time
print("Elapsed time: %0.10f seconds." % elapsed_time)

# =============================================================================
# Prueba PSF simetrica sin torch
# =============================================================================
# def sersic(n,theta,r_eff,ellip,x,x_0,y,y_0,amplitude):
#         bn = (2.0*n) - (0.327)
#         theta = theta*np.pi/180
#         a, b = r_eff, (1-ellip) * r_eff
#         cos_theta, sin_theta = np.cos(theta), np.sin(theta)
#         x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
#         x_maj = -(x - x_0) * cos_theta - (y - y_0) * sin_theta

#         z = np.sqrt(((x_min / a)**2) + ((x_maj/b)**2))
#         return(amplitude * np.exp(-bn*(((z**(1/n))) - 1)))

# y, x = torch.meshgrid(torch.arange(int(ny)), torch.arange(int(nx)))
# modelin = sersic(5.1032339340992072,84.6835058750300078,80.0539328733860458,0,x,182.67604,y,154.94841,24.5104490606881136*gain)
# modelin=modelin.detach().numpy()
# psf_class = psf(51,51,psf_imgname,gauss_amp,mean_x, mean_y, theta_rot,\
#                 stdv_x,stdv_y,moff_amp,25, 25,width_moff,power_moff)
# class_method = getattr(psf_class, psf_type)
# psf_image = class_method()
# data_conv = convolve(modelin, psf_image)
# residual = data_conv-modelin
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
# mapi =ax[0].imshow(modelin,cmap = cmap)
# cbar=fig.colorbar(mapi,ax=ax[0],shrink=0.5,extend='both')
# cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
# ax[0].set_title("Model")
# mapi = ax[1].imshow( data_conv,cmap = cmap)
# cbar=fig.colorbar(mapi,ax=ax[1],shrink=0.5,extend='both')
# cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
# ax[1].set_title("Model+Conv")
# mapi = ax[2].imshow(residual,vmin=-1000,cmap = cmap)
# cbar=fig.colorbar(mapi,ax=ax[2],shrink=0.5,extend='both')
# cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
# ax[2].set_title("Residual map")
# plt.tick_params(axis="x", direction="in", length=7, width=1.2, color="k")
# plt.tick_params(axis="y", direction="in", length=7, width=1.2, color="k")
# ax[0].set_ylabel(r'y [px]', fontsize = 16)
# ax[0].set_xlabel(r'x [px]', fontsize =16)
# ax[0].xaxis.set_minor_locator(AutoMinorLocator())
# ax[0].yaxis.set_minor_locator(AutoMinorLocator())
# ax[0].tick_params(direction="in",which='minor', length=4, color='k')
# ax[1].set_ylabel(r'y [px]', fontsize = 16)
# ax[1].set_xlabel(r'x [px]', fontsize =16)
# ax[1].xaxis.set_minor_locator(AutoMinorLocator())
# ax[1].yaxis.set_minor_locator(AutoMinorLocator())
# ax[1].tick_params(direction="in",which='minor', length=4, color='k')
# ax[2].set_ylabel(r'y [px]', fontsize = 16)
# ax[2].set_xlabel(r'x [px]', fontsize =16)
# ax[2].xaxis.set_minor_locator(AutoMinorLocator())
# ax[2].yaxis.set_minor_locator(AutoMinorLocator())
# ax[2].tick_params(direction="in",which='minor', length=4, color='k')

# =============================================================================
# Prueba PSF simetrica con torch
# =============================================================================
# paramis = np.array([24.5104490606881136*gain,80.0539328733860458,5.1032339340992072,100.1,100.1,1- 0.7658242620261781,84.6835058750300078])
# params2=torch.tensor(paramis)
# model_class = profiles(x_size=data.shape[1],y_size=data.shape[0])
# model_method = model_class.Sersic(amp_sersic=params2[0],\
#                                   r_eff_sersic=params2[1], n_sersic=params2[2]\
#                                       ,x0_sersic=params2[3], \
#                                           y0_sersic=params2[4],\
#                                               ellip_sersic=params2[5], \
#                                                   theta_sersic=params2[6])
# ma = model_method
# b =  (hamiltonian_class.conv2d_fft_psf(ma))

# residual = b.detach().numpy()-ma.detach().numpy()
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
# mapi =ax[0].imshow((ma.detach().numpy()),cmap = cmap)
# cbar=fig.colorbar(mapi,ax=ax[0],shrink=0.5,extend='both')
# cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
# ax[0].set_title("Model")
# mapi = ax[1].imshow( b.detach().numpy(),cmap = cmap)
# cbar=fig.colorbar(mapi,ax=ax[1],shrink=0.5,extend='both')
# cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
# ax[1].set_title("Model + Conv")
# mapi = ax[2].imshow(residual,vmin=-2000,cmap = cmap)
# cbar=fig.colorbar(mapi,ax=ax[2],shrink=0.5,extend='both')
# cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
# ax[2].set_title("Residual map")
# plt.tick_params(axis="x", direction="in", length=7, width=1.2, color="k")
# plt.tick_params(axis="y", direction="in", length=7, width=1.2, color="k")
# ax[0].set_ylabel(r'y [px]', fontsize = 16)
# ax[0].set_xlabel(r'x [px]', fontsize =16)
# ax[0].xaxis.set_minor_locator(AutoMinorLocator())
# ax[0].yaxis.set_minor_locator(AutoMinorLocator())
# ax[0].tick_params(direction="in",which='minor', length=4, color='k')
# ax[1].set_ylabel(r'y [px]', fontsize = 16)
# ax[1].set_xlabel(r'x [px]', fontsize =16)
# ax[1].xaxis.set_minor_locator(AutoMinorLocator())
# ax[1].yaxis.set_minor_locator(AutoMinorLocator())
# ax[1].tick_params(direction="in",which='minor', length=4, color='k')
# ax[2].set_ylabel(r'y [px]', fontsize = 16)
# ax[2].set_xlabel(r'x [px]', fontsize =16)
# ax[2].xaxis.set_minor_locator(AutoMinorLocator())
# ax[2].yaxis.set_minor_locator(AutoMinorLocator())
# ax[2].tick_params(direction="in",which='minor', length=4, color='k')