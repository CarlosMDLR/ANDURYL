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
#  Model plot
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

fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(15, 15))
mapi =ax[0].imshow((data*norm_I),cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[0],shrink=0.5,extend='both')
cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[0].set_title("Data")
mapi = ax[1].imshow( b.detach().numpy()*norm_I,cmap = cmap)
cbar=fig.colorbar(mapi,ax=ax[1],shrink=0.5,extend='both')
cbar.set_label(r"I [$e^{-}$]",loc = 'center',fontsize = 16)
ax[1].set_title("Model")
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
plt.savefig('\Documentos\Master Astrofisica\TFM\ANDURYL\Figures'+ '\%s'%("figura_3"), bbox_inches='tight', pad_inches=0.02)

