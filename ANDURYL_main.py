""" """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ 
===============================================================================

                                    ANDURYL
            A bayesiaN Decomposition code for Use in photometRY Labors
                                    MADE BY:
                           -Carlos Marrero de la Rosa
                           -Andres Asensio Ramos
                           -Jairo Mendez-Abreu
                         FORGED IN THE YEARS 2022-2023
                         
===============================================================================

                                    .*,,,,,,                                                
                                  ,***/((#&,,,                                              
                                   /*/**(,,/(,                                              
                                    ***/**,,,                                               
                                      ***,,                                                 
                                      ***,,                                                 
                                      ***,,                                                 
                               *@@@@@&****,@@@@(                                            
                        .@@@@@@@@##@@&*(&**@%%@@@@@@@@                                      
                    *@@@@@@@//&@@@&%@&/#&**@%%@@@%#@@@@@@@                                  
                 /@@@@@#@/@@@/&@@@#%@&/&&**%%@@@%#@@@%@@@@@@@                               
               @@@@@@@@@@/&%@@/@@@@@@&/////@@@@@%@@@(@@@@@@@@@@%                            
             @@@@@@@@@@@@@@@@@@@@(   &&&&&&  .&@@@@@@@@@@@@@@@@@@%                          
           .@@@@@@@@@@@@@@@@@        @@@&&&.      ,@@@@@@@@@@@@@@@@                         
          (@@@@@@@@@@@@@@@           @@@&&&          (@@@@@@@@@@@@@@#                       
         *@@@@@@@@@@@@@@             @@@@&&            %@@@@@@@@@@@@@%                      
         @@@@@@@@@@@@@@              @@@@&&             .@@@@@@@@@@@@@,                     
        #@@@@@@@@@@@@@               @@@@&&              *@@@@@@@@@@@@@                     
        @@@@@@@@@@@@@                @@@@&&           #&&&&&&&&@@@@@@@@                     
        @@@@@@@@@@@@@                @@@@&&          &&#  @@@%%&%@@@@@@/                    
        @@@@@@@@@@@@@.  ,(#%%%%#(/*. @@@@@&/%*       #&   @@@&##%&@@@@@,                    
        *@@@@@@@@@@##(((((////(/,,   ,#*(%%#///(((       &@@@####&@@@@@                     
         @@@@@@@@%#(((##(///****,,,,,//@@@&#*.,*//***/((#(**//(#%@@@@@&                     
          @@@@@@%#(##@@@ ,     .(/((((#/(%&#*,,*////*,**/*,*/#@@@@@@@@                      
           @@@@@&%%%&@@@@@.&        (//**,,,,    ..,,.@@@@@@@@@@@@@@@                       
            @@@@&&&&%%@@@@&&(       ((/**,,,,      @@@@@@@@@@@@@@@@@                        
             #@@@@#&&&&&@@@@@@@@@(  ((//*,,,, *@@@@@@@@@@@@@@@@@@@.                         
               &@@@@@@@@@@@@@@@@@@@@(///**,,,@@@@@@@@@@@@@@@@@@@*                           
                 ,@@@@@@@@@@@@@@@@@@(//*,*,,,@@@@@@@@@@@@@@@@@                              
                    /@@@@@@@@@@@@@@@(//***,,,@@@@@@@@@@@@@@,                                
                        #@@@@@@@@@@@(//*,*,,,@@@@@@@@@@%                                    
                              %@@@@@(//*,*,,,@@@@@,                                         
                                    ///*,*,,,                                               
                                    ///*,*,,,                                               
                                    ///*,*,,,                                               
                                    ///*,*,,,                                               
                                    ////,*,,.                                               
                                    (///,*,,,                                               
                                    (///,*,,,                                               
                                    (///**,,,                                               
                                     ///**,,,                                               
                                      ./**,,,                                               
                                       /**,,,                                               
                                       /**,,,                                               
                                       /**,,,                                               
                                        */,,,                                               
                                         /*,,                                               
                                          *,,                                               
                                          ,,,                                               
                                          ,,,                                               
                                           ,,                                               
                                           ,,                                               
                                            ,                                               
                                            ,                                               
                                            . 
                                            
===============================================================================

        This code has been designed with the aim of performing photometric 
        decompositions of galaxies using Bayesian statistics with different 
        inference methods.  
                         
===============================================================================                                            
""" """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """
# =============================================================================
# Import modules
# =============================================================================
import sys
import torch
import corner
import os
import numpy as np
import pandas as pd
import cblind as cb
#import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from time import time
from astropy.io import fits

#from getdist import plots,MCSamples
from joblib import Parallel, delayed
from matplotlib.ticker import (AutoMinorLocator)

modulos_folder = './Python_Modules/'                                        
modulos_path = modulos_folder.replace('.', os.getcwd().replace('\\', '/'))  
if modulos_path not in sys.path:
    sys.path.append(modulos_path)

import read_params as pars
from psf_generation import psf
from profile_select import profiles
from hamiltonian_class import hamiltonian_model

# Start counting the computing time
start_time = time()

# =============================================================================
# Functions
# =============================================================================

def compare_plots(mn,hamiltonian_class,samples,data,labels,titles,\
                  colorbarname,path,name,log,gain,nmgy):
    """
     This function is used to make the plots of the galaxies 
     and compare them with the generated models and obtain the residuals
    """

    #  Model calculation
    if pars.to_fit=="Bulge-disc":
        # When fitting a disk bulb, both types of profiles
        # are called to generate the model
        model_class = profiles(x_size=data.shape[1],y_size=data.shape[0])
        model_method_sersic = model_class.Sersic(amp_sersic=mn[0],\
                                          r_eff_sersic=mn[1], n_sersic=mn[2]\
                                              ,x0_sersic=mn[3], \
                                                  y0_sersic=mn[4],\
                                                      ellip_sersic=mn[5], \
                                                          theta_sersic=mn[6])
        model_method_exp = model_class.Exponential(amp_expo=mn[9],\
                                        r_eff_expo=mn[10], x0_expo=mn[3],\
                                            y0_expo=mn[4], ellip_expo=mn[11],\
                                            theta_expo=mn[12])
        ma = model_method_sersic+model_method_exp

        # The image of the PDF is generated
        psf_class = psf(pars.xsize,pars.ysize,pars.gauss_amp,pars.mean_x,\
                        pars.mean_y, pars.theta_rot,pars.stdv_x,pars.stdv_y,\
                            pars.moff_amp,pars.xsize//2,pars.ysize//2,mn[7],\
                                mn[8],None, None, None, None,None,None,None)
        class_method = getattr(psf_class, pars.psf_type)
        psf_image = class_method() 
        # Here we proceed to carry out previous steps for the convolution with 
        # the psf. You have to do zero padding and be careful where python 
        # starts to perform the convolution. That is why they are made 
        # iffftshift now
        sz = (data.shape[0] - psf_image.shape[0], data.shape[1] - \
              psf_image.shape[1])
        kernel = np.pad(psf_image, (((sz[0]+1)//2, sz[0]//2),\
                                    ((sz[1]+1)//2,sz[1]//2)))
        psf_mod = torch.tensor(kernel)
        psf_fft = torch.fft.fft2( torch.fft.ifftshift(psf_mod))
        # The model is convolved with the PDF
        b = hamiltonian_class.conv2d_fft_psf(ma,psf_fft)
        # The generated model is saved (in counts) in a fits file
        if not log:
            hdu = fits.PrimaryHDU(b/gain)

            hdu.writeto(path+"/bulge_disc_"+ name[:-5]+'.fits',overwrite=True)
    elif pars.to_fit=="Bulge":
        # A model is generated only from a Sersic profile
        model_class = profiles(x_size=data.shape[1],y_size=data.shape[0])
        model_method = model_class.Sersic(amp_sersic=mn[0],\
                                          r_eff_sersic=mn[1], n_sersic=mn[2]\
                                              ,x0_sersic=mn[3], \
                                                  y0_sersic=mn[4],\
                                                      ellip_sersic=mn[5], \
                                                          theta_sersic=mn[6])
        ma = model_method
        # The image of the PDF is generated
        psf_class = psf(pars.xsize,pars.ysize,pars.gauss_amp,pars.mean_x, \
                        pars.mean_y, pars.theta_rot,pars.stdv_x,pars.stdv_y,\
                            pars.moff_amp,pars.xsize//2,pars.ysize//2,mn[7],\
                                mn[8],None, None, None, None,None,None,None)
        class_method = getattr(psf_class, pars.psf_type)
        psf_image = class_method() 
        # Here we proceed to carry out previous steps for the convolution with 
        # the psf. You have to do zero padding and be careful where python 
        # starts to perform the convolution. That is why they are made 
        # iffftshift now
        sz = (data.shape[0] - psf_image.shape[0], data.shape[1] - \
              psf_image.shape[1])
        kernel = np.pad(psf_image, (((sz[0]+1)//2, sz[0]//2),\
                                    ((sz[1]+1)//2,sz[1]//2)))
        psf_mod = torch.tensor(kernel)
        psf_fft = torch.fft.fft2( torch.fft.ifftshift(psf_mod))
        # The model is convolved with the PDF
        b = hamiltonian_class.conv2d_fft_psf(ma,psf_fft)
        # The generated model is saved (in counts) in a fits file
        if not log:
            hdu = fits.PrimaryHDU(b/gain)

            hdu.writeto(path+"/bulge_"+ name[:-5]+'.fits',overwrite=True)
    # =========================================================================
    # Data-model-residual plot
    # =========================================================================
    if not log:
        data = abs(data)/gain
        b = b/gain            
        residual = abs(data)-b.detach().numpy()
    elif log:
        # We transform the result from counts to superficial bright

        if not use_simu:
            data = -2.5*np.log10(abs(data)/gain*nmgy)+20.48847593
            b=-2.5*torch.log10(b/gain*nmgy)+20.48847593
        elif use_simu:
            data = -2.5*np.log10(abs(data)/gain)+20.48847593
            b=-2.5*torch.log10(b/gain)+20.48847593
        residual = abs(data)-b.detach().numpy()

    # We generate the figures
    fig = plt.figure(figsize=(15,25))

    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[0, :2], )
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, 1:3])

    plt.rc('xtick', labelsize=24)    
    plt.rc('ytick', labelsize=24) 
    if not log:
        # If the units are in counts, the maximum emitted by the galaxy is 
        # set as a limit, which, given the nature of the images, should be more
        # or less in a square around the center of the image.
        box = data[int(mn[4]-15):int(mn[4]+15),int(mn[3]-15):int(mn[3]+15)]

        maxi = box.max()
        mapi =ax1.imshow((abs(data)),vmax = maxi,cmap = cmap)
    if log:
        mapi =ax1.imshow((data),cmap = cmap)
    # Here are the color bars and the different axes are labeled
    cbar=fig.colorbar(mapi,ax=ax1,shrink=0.9,extend='both')
    cbar.set_label(colorbarname,loc = 'center',fontsize = 22)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax1.set_title(titles["Data"], fontsize = 26)
    mapi = ax2.imshow( b.detach().numpy(),cmap = cmap)
    cbar=fig.colorbar(mapi,ax=ax2,shrink=0.9,extend='both')
    cbar.set_label(colorbarname,loc = 'center',fontsize = 22)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax2.set_title(titles["Model"], fontsize = 26)
    if not log:
        if use_simu:
            mapi = ax3.imshow(residual,cmap = cmap)
        if not use_simu:
            mapi = ax3.imshow(residual,vmin=-(1/3)*maxi,cmap = cmap)
        cbar=fig.colorbar(mapi,ax=ax3,shrink=0.9,extend='both')
        cbar.set_label(colorbarname,loc = 'center',fontsize = 22)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
    if log:
        mapi = ax3.imshow(residual,vmin=-0.5,vmax=0.5,cmap = cmap)
        cbar=fig.colorbar(mapi,ax=ax3,orientation='vertical',shrink=0.9,\
                          label='Some Units', extend='both')
        cbar.set_label(colorbarname,loc = 'center',fontsize = 22)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
    ax3.set_title(titles["Residual map"], fontsize = 26)
    plt.tick_params(axis="x", direction="in", length=7, width=1.2, color="k")
    plt.tick_params(axis="y", direction="in", length=7, width=1.2, color="k")
    ax1.set_ylabel(labels["y"], fontsize = 26)
    ax1.set_xlabel(labels["x"], fontsize =26)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(direction="in",which='minor', length=4, color='k')
    ax1.locator_params(axis='both', nbins=5)
    ax2.set_ylabel(labels["y"], fontsize = 26)
    ax2.set_xlabel(labels["x"], fontsize =26)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(direction="in",which='minor', length=4, color='k')
    ax2.locator_params(axis='both', nbins=5)
    ax3.set_ylabel(labels["y"], fontsize = 26)
    ax3.set_xlabel(labels["x"], fontsize =26)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.tick_params(direction="in",which='minor', length=4, color='k')
    ax3.locator_params(axis='both', nbins=5)
    #The images are saved in the corresponding files
    if not log:
        if pars.to_fit=="Bulge":
            plt.savefig(path+"/bulge_"+ name[:-5],\
                        bbox_inches='tight',pad_inches=0.02)
            plt.close()
        elif pars.to_fit=="Bulge-disc":
            plt.savefig(path+"/bulge_disc_"+ name[:-5],\
                        bbox_inches='tight', pad_inches=0.02)
            plt.close()
    elif log:
        if pars.to_fit=="Bulge":
            plt.savefig(path+"/log_bulge_"+ name[:-5],\
                        bbox_inches='tight', pad_inches=0.02)
            plt.close()
        elif pars.to_fit=="Bulge-disc":
            plt.savefig(path+"/log_bulge_disc_"+ name[:-5],\
                        bbox_inches='tight', pad_inches=0.02)
            plt.close()      
    return()


def trianguar_plots_corner(mn,samples,path,name,gain):
    """
    This function performs the triangular diagrams
    """
    plt.rc('xtick', labelsize=16)    
    plt.rc('ytick', labelsize=16) 

        
    if pars.to_fit=="Bulge": 
        samples[:,0]/=gain
        mn[0]/=gain
        # Samples are loaded when adjusting a Sersic profile
        # the names of the adjusted parameters are entered as well 
        # as some aesthetic parameters of the plots
        fig=corner.corner(samples,\
        var_names=['I0','Re','n','x0','y0','varepsilon','PA','FWHM',\
                   'Beta','Noise'],\
        labels=[r'$I_0$ [Counts]', r'$R_e$ [px]',r'$n$',r'$x_0$ [px]',\
                r'$y_0$ [px]',r'$\varepsilon$',r'$PA$ [º]','FWHM [px]',\
                r"$\beta$","Noise"],\
        show_titles=True,max_n_ticks=3,\
        title_kwargs={"fontsize":14},\
        label_kwargs=dict(fontsize=14),\
        plot_datapoints=True,fill_contours=True,use_math_text=True)
    
    elif pars.to_fit=="Bulge-disc":
 
        samples[:,0]/=gain
        mn[0]/=gain
        samples[:,10]/=gain
        mn[10]/=gain

        # Samples are loaded when adjusting a Sersic+exponential profile
        # the names of the adjusted parameters are entered as well 
        # as some aesthetic parameters of the plots
        fig=corner.corner(samples,\
        var_names=['I0','Re','n','x0','y0','varepsilon','PA', 'FWHM',\
                   'Beta','Noise','Ie','Re_exp','varepsilon_exp','PA_exp'],\
        labels=[r'$I_0^{\mathrm{sersic}}$[Counts]',r'$R_e^{\mathrm{sersic}}$[px]',\
                r'$n$',r'$x_0$ [px]',r'$y_0$ [px]',\
                r'$\varepsilon^{\mathrm{sersic}}$',\
                r'$PA^{\mathrm{sersic}}$ [º]', 'FWHM [px]',\
                r"$\beta$","Noise",r'$I_0^{\mathrm{exp}}$ [Counts]',\
                r'$R_e^{\mathrm{exp}}$ [px]',r'$\varepsilon^{\mathrm{exp}}$',\
                r'$PA^{\mathrm{exp}}$ [º]'],\
        show_titles=True,max_n_ticks=3,\
        title_kwargs={"fontsize":14},\
        label_kwargs=dict(fontsize=14),\
        plot_datapoints=True, fill_contours=True,use_math_text=True) 
    # The figure is created and the minorticks and the lines for the mean of
    # each parameter are added.
    fig.subplots_adjust(right=1.5,top=1.5)
    corner.overplot_lines(fig, mn.detach(), color="C2")
    corner.overplot_points(fig, mn.detach()[None], marker="o", color="C2")
    for ax in fig.axes:
        plt.tick_params(axis="x",direction="in",length=7,width=1.2,color="k")
        plt.tick_params(axis="y",direction="in",length=7,width=1.2,color="k")

        ax.xaxis.set_tick_params(rotation=35)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction="in",which='minor', length=4, color='k')
    # Triangular plots are saved according to the type of adjustment
    # that has been made
    if pars.to_fit=="Bulge":
        fig.savefig(path+"/bulge_triangular_"+ name[:-5],dpi=300,\
                    pad_inches=0.04,bbox_inches='tight')
        plt.close()
    elif pars.to_fit=="Bulge-disc":
        fig.savefig(path+"/bulge_disc_triangular_"+ name[:-5],dpi=300,\
                    pad_inches=0.04,bbox_inches='tight')
        plt.close()
    return

def iter_gal_parallel(gal_files,frame_names,psf_values,objID,run,camcol,field,\
                      filte,length_run,length_field,mask_files,index,\
                      width_moff,power_moff,gain,sky_val,readout_noise,\
                      dir_img_final):
    """
    This function is the one that is created so that the adjustments are 
    made through the Bayesian method.
    """
    if comm_params:
        # This is in case mock galaxies are analyzed
        # The image is charged and passed to electrons
        data=fits.getdata(pars.gal_dir+'/'+gal_files[int(index)])*gain
        # PSF parameters are loaded
        width_moff=pars.width_moff
        power_moff=pars.power_moff
        nmgy_percount=None
    if not comm_params:
        # This is in case you work with real galaxies from SDSS
        index_2 = np.where((objID ==int(gal_files[int(index)][:-5])))[0][0]
        # Here you select the frame in which each galaxy is located
        count_run = length_run-len(str(run[index_2]))
        run_new=count_run*'0'+str(run[index_2])
        count_field = length_field-len(str(field[index_2]))
        field_new=count_field*'0'+str(field[index_2])
        frame_name = "frame-%s"%filte +"-%s"%run_new+"-%s"%\
            (str(camcol[index_2]))+"-%s"%field_new+".fits"
        
        # SDSS filter numbering
        if filte=="r":ind_filt=0
        if filte=="i":ind_filt=1
        if filte=="u":ind_filt=2
        if filte=="z":ind_filt=3
        if filte=="g":ind_filt=4
        
        index_3 = np.where((frame_names==frame_name))[0][0]
        
        try:
            width_moff=psf_values[index_3][0]
            power_moff=psf_values[index_3][1]
        except:
            width_moff=psf_values[index_3]
            power_moff=psf_values[index_3]            
        # SDSS PhotField data is loaded
        phot_field_data = fits.getdata(pars.file_params+"/"+"photoField"+"-%s"\
                                       %run_new+"-%s"%(str(camcol[index_2]))\
                                       +".fits")
        
        gain = np.mean(phot_field_data["Gain"][:,ind_filt])
        sky_val = np.mean(phot_field_data["Sky"][:,ind_filt])
        readout_noise = np.mean(phot_field_data["Dark_variance"][:,ind_filt])
        nmgy_percount = np.mean(phot_field_data["NMGYPERCOUNT"][:,ind_filt])
        
        # Different parameters of each galaxy are loaded, such as the gain
        # value, the sky value and the conversion factor from nanomaggies 
        # to counts.
        data=fits.getdata(pars.gal_dir+'/'+gal_files[int(index)])\
            *gain/nmgy_percount
        
        pars.sky_sigm= np.mean(phot_field_data["Skysig"][:,ind_filt])
    
    # The masks of each galaxy are loaded
    if mask_files is None:
        mask = np.zeros((data.shape[0],data.shape[1]))
    if mask_files is not None:
        mask = fits.getdata(pars.mask_dir+'/'+ mask_files[index])
    
    # Noise standard deviation map in counts
    if comm_params:
        noise_map = np.sqrt(np.abs(data) + (sky_val * gain) + readout_noise**2)
    if not comm_params:
        noise_map = np.sqrt(np.abs(data) + (sky_val * gain/nmgy_percount) + readout_noise**2)
  
    # =========================================================================
    # Application of the bayesian method
    # =========================================================================
    hamiltonian_class = hamiltonian_model(data.astype(np.float64),\
                                          width_moff,power_moff,mask,gain,\
                                          noise_map,dir_img_final,\
                                          gal_files[int(index)])
    samples = hamiltonian_class.hamiltonian_sampling()   
    return(samples,hamiltonian_class,data,gal_files[int(index)],gain,nmgy_percount)

# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# Creation of directories to store different types of data
# =============================================================================
use_simu = str(pars.use_simu).lower() in ['yes']
comm_params = str(pars.comm_params).lower() in ['yes']
if not os.path.isdir(pars.dir_img_final): os.makedirs(pars.dir_img_final)
if use_simu:
    # Directories to store the information of the mock galaxies
    if not os.path.isdir(pars.dir_img_final+"/Images_simu"):\
        os.makedirs(pars.dir_img_final+"/Images_simu")
    if not os.path.isdir(pars.dir_img_final+"/Log_Images_simu"):\
        os.makedirs(pars.dir_img_final+"/Log_Images_simu")
    if not os.path.isdir(pars.dir_img_final+"/Triangular_plots_simu"):\
        os.makedirs(pars.dir_img_final+"/Triangular_plots_simu")
    if pars.to_fit=="Bulge":
        if not os.path.isdir(pars.dir_img_final+"/simulation_samples_bulge"):\
            os.makedirs(pars.dir_img_final+"/simulation_samples_bulge")
    elif pars.to_fit=="Bulge-disc":
        if not os.path.isdir(pars.dir_img_final+\
                             "/simulation_samples_bulge_disc"):\
            os.makedirs(pars.dir_img_final+"/simulation_samples_bulge_disc")
if not use_simu:
    # Directories to store information on real galaxies
    if not os.path.isdir(pars.dir_img_final+"/Images_gals"):\
        os.makedirs(pars.dir_img_final+"/Images_gals")
    if not os.path.isdir(pars.dir_img_final+"/Log_Images_gals"):\
        os.makedirs(pars.dir_img_final+"/Log_Images_gals")
    if not os.path.isdir(pars.dir_img_final+"/Triangular_plots_gals"):\
        os.makedirs(pars.dir_img_final+"/Triangular_plots_gals")
    if pars.to_fit=="Bulge":
        if not os.path.isdir(pars.dir_img_final+"/gals_samples_bulge"):\
            os.makedirs(pars.dir_img_final+"/gals_samples_bulge")
    elif pars.to_fit=="Bulge-disc":
        if not os.path.isdir(pars.dir_img_final+"/gals_samples_bulge_disc"):\
            os.makedirs(pars.dir_img_final+"/gals_samples_bulge_disc")

# =============================================================================
# Reading data
# =============================================================================

if use_simu:
    galaxies,Ie,Re,n, ba_b, PA_bulge, B_T,Mag_tot=\
        pars.data_reader(pars.user_in_file)

# The files are loaded from the directory chosen to analyze

gal_files = [ f for f in os.listdir(pars.gal_dir) if f.endswith(".fits")]
gal_files.sort()

# The masks are loaded, not having any in the case of the mock galaxies
if use_simu:
    mask_files=None
if not use_simu:
    mask_files = [ f for f in os.listdir(pars.mask_dir) if f.endswith(".fits")]
    mask_files.sort()

# The values of the names of the frames and the values of the PSF are loaded,
# which in the case of mock galaxies will come by default in the
# parameters file. Different parameters are also loaded for real galaxies

if  comm_params:
    frame_names=None
    psf_values=None
    
    csv_file = None
    objID=None
    run=None
    camcol=None
    field=None
    rowc_i=None
    colc_i=None
    
    filte=None
    length_run= None
    length_field=None
if not comm_params:
    frame_names=np.loadtxt(pars.params_dir,skiprows=1,usecols=(0),dtype=str)
    psf_values=np.loadtxt(pars.params_dir,skiprows=1,usecols=(1,2))
    

    csv_file = pd.read_csv(pars.gal_csv,header=1)
    objID=csv_file["objID" ]
    run=(csv_file["run" ])
    camcol=(csv_file["camcol" ])
    field=(csv_file["field" ])
    rowc_i=(csv_file["rowc_i" ])
    colc_i=(csv_file["colc_i" ])
    
    filte="i"
    length_run= 6
    length_field=4

# =============================================================================
# Perform model fit to galaxy
# =============================================================================

# A parallelization process is carried out
n_jobs=int(pars.n_cores)
gal_lim = np.arange(int(pars.low_lim),int(pars.upp_lim),1)

result = Parallel(n_jobs=n_jobs,verbose =10)(delayed(iter_gal_parallel)
                        (gal_files,frame_names,psf_values,objID,run,camcol,\
                         field,filte,length_run,length_field,mask_files,index,\
                         pars.width_moff,pars.power_moff,pars.gain,\
                         pars.sky_val,pars.readout_noise,pars.dir_img_final)
                        for index in gal_lim)
# =============================================================================
# Make all plots
# =============================================================================
if pars.make_plots:
    #cmap = plt.get_cmap('hot') #selection of color map
    #This is just a colormap (suitable for color blinds)
    cmap= plt.get_cmap("cb.iris_r")
    for i in range(0,len(gal_lim)):
        if result[i][0] is not None:
            mn = torch.tensor(np.median(result[i][0], axis=0))
            if use_simu:
                compare_plots(mn,result[i][1],result[i][0],result[i][2],\
                              {"x":"x","y":"y"},\
                              {"Data":"Data","Model":"Model",\
                               "Residual map": "Residual map"},\
                                r"I [Counts]",pars.dir_img_final\
                                    +"/Images_simu",result[i][3],False\
                                        ,result[i][4],result[i][5])
                compare_plots(mn,result[i][1],result[i][0],result[i][2],\
                              {"x":"x","y":"y"},\
                              {"Data":"Data","Model":"Model",\
                               "Residual map": "Residual map"},\
                                r"$\mu [mag/arcsec^2]$",pars.dir_img_final\
                                +"/Log_Images_simu",result[i][3],True\
                                    ,result[i][4],result[i][5])
                trianguar_plots_corner(mn,result[i][0],pars.dir_img_final\
                                       +"/Triangular_plots_simu",result[i][3]\
                                           ,result[i][4])
            if not use_simu:
                compare_plots(mn,result[i][1],result[i][0],result[i][2],\
                              {"x":"x","y":"y"},\
                              {"Data":"Data","Model":"Model",\
                               "Residual map": "Residual map"},\
                                r"I [Counts]",pars.dir_img_final\
                                    +"/Images_gals",result[i][3],False\
                                        ,result[i][4],result[i][5])
                compare_plots(mn,result[i][1],result[i][0],result[i][2],\
                              {"x":"x","y":"y"},\
                              {"Data":"Data","Model":"Model",\
                               "Residual map": "Residual map"},\
                                r"$\mu [mag/arcsec^2]$",pars.dir_img_final\
                                    +"/Log_Images_gals",result[i][3],True\
                                        ,result[i][4],result[i][5])
                trianguar_plots_corner(mn,result[i][0],pars.dir_img_final\
                                       +"/Triangular_plots_gals",result[i][3]\
                                           ,result[i][4])
        else:
            continue
# =============================================================================
# Save results of each parameter
# =============================================================================
results_list = []
name_list = []
for i in range(0,len(gal_lim)):
    if result[i][0] is not None:
        mn = np.median(result[i][0], axis=0)
        results_list.append(mn.tolist())
        name_list.append(result[i][3][:-5])
    else:
        continue

final_list_2 = [[name] + values for name, values in zip(name_list, results_list)]
if pars.to_fit=="Bulge":
    df = pd.DataFrame(final_list_2,\
                      columns = ['#Gal_Name','I0[e-]','Re[px]','n','x0[px]',\
                                 'y0[px]','varepsilon','PA[º]',\
                                 'FWHM[px]','Beta','Noise'])
elif pars.to_fit=="Bulge-disc":
    df = pd.DataFrame(final_list_2,\
                      columns = ['#Gal_Name','I0[e-]','Re[px]','n','x0[px]',\
                                 'y0[px]','varepsilon','PA[º]',\
                                 'FWHM[px]','Beta','Noise','Ie[e-]',\
                                 'Re_exp[px]','varepsilon_exp',\
                                 'PA_exp[º]'])
df.to_csv(pars.dir_img_final+'/test.txt', index = False, sep=' ')

# Calculate the total elapsed time
elapsed_time = time() - start_time
print("Elapsed time: %0.10f seconds." % elapsed_time)
