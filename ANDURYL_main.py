# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:27:44 2022
@author: Usuario
"""

"""
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
from joblib import Parallel, delayed
import os,glob
import pandas as pd
import corner

# Start counting
start_time = time()


# =============================================================================
# Functions
# =============================================================================

def compare_plots(mn,hamiltonian_class,samples,data,labels,titles,colorbarname,path,name,log):

    # =============================================================================
    #  Model calculation
    # =============================================================================

    model_class = profiles(x_size=data.shape[1],y_size=data.shape[0])
    model_method = model_class.Sersic(amp_sersic=mn[0],\
                                      r_eff_sersic=mn[1], n_sersic=mn[2]\
                                          ,x0_sersic=mn[3], \
                                              y0_sersic=mn[4],\
                                                  ellip_sersic=mn[5], \
                                                      theta_sersic=mn[6])
    ma = model_method
    b = hamiltonian_class.conv2d_fft_psf(ma)
    
    if not log:
        hdu = fits.PrimaryHDU(b)
        hdu.writeto(path+"/"+ name[:-5]+'.fits',overwrite=True)
    # =============================================================================
    # Data-model-residual plot
    # =============================================================================
    if not log:
        residual = b.detach().numpy()-data
    elif log:
        residual = b.detach().numpy()-data
        data=-2.5*np.log10((data))+23.602800
        b=-2.5*torch.log10(b)+23.602800
        residual = (abs(residual)/residual)*(-2.5*np.log10(abs(residual))+23.602800)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 15))
    if not log:
        mapi =ax[0].imshow((data),vmax=10000,cmap = cmap)
    if log:
        mapi =ax[0].imshow((data),cmap = cmap)
    cbar=fig.colorbar(mapi,ax=ax[0],shrink=0.5,extend='both')
    cbar.set_label(colorbarname,loc = 'center',fontsize = 16)
    ax[0].set_title(titles["Data"])
    mapi = ax[1].imshow( b.detach().numpy(),cmap = cmap)
    cbar=fig.colorbar(mapi,ax=ax[1],shrink=0.5,extend='both')
    cbar.set_label(colorbarname,loc = 'center',fontsize = 16)
    ax[1].set_title(titles["Model"])
    if not log:
        mapi = ax[2].imshow(residual,vmin=-10000,cmap = cmap)
    if log:
        mapi = ax[2].imshow(residual,cmap = cmap)
    cbar=fig.colorbar(mapi,ax=ax[2],shrink=0.5,extend='both')
    cbar.set_label(colorbarname,loc = 'center',fontsize = 16)
    ax[2].set_title(titles["Residual map"])
    plt.tick_params(axis="x", direction="in", length=7, width=1.2, color="k")
    plt.tick_params(axis="y", direction="in", length=7, width=1.2, color="k")
    ax[0].set_ylabel(labels["y"], fontsize = 16)
    ax[0].set_xlabel(labels["x"], fontsize =16)
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].tick_params(direction="in",which='minor', length=4, color='k')
    ax[1].set_ylabel(labels["y"], fontsize = 16)
    ax[1].set_xlabel(labels["x"], fontsize =16)
    ax[1].xaxis.set_minor_locator(AutoMinorLocator())
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].tick_params(direction="in",which='minor', length=4, color='k')
    ax[2].set_ylabel(labels["y"], fontsize = 16)
    ax[2].set_xlabel(labels["x"], fontsize =16)
    ax[2].xaxis.set_minor_locator(AutoMinorLocator())
    ax[2].yaxis.set_minor_locator(AutoMinorLocator())
    ax[2].tick_params(direction="in",which='minor', length=4, color='k')
    
    if not log:
        plt.savefig(path+"/"+ name[:-5], bbox_inches='tight', pad_inches=0.02)
        plt.close()
    elif log:
        plt.savefig(path+"/log_"+ name[:-5], bbox_inches='tight', pad_inches=0.02)
        plt.close()        
    return()

def trianguar_plots(mn,samples,path,name):
    g = plots.get_subplot_plotter(width_inch=24)
    g.settings.lab_fontsize = 14
    g.settings.axes_fontsize = 14
    g.settings.num_plot_contours = 4
    g.settings.axis_tick_x_rotation=45
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.4
    g.settings.legend_fontsize=16
    g.settings.axis_tick_max_labels=3
    g.settings.title_limit_fontsize = 13
    g.settings.prob_y_ticks=True
    g.settings.solid_contour_palefactor = 0.9
    g.settings.alpha_filled_add = 0.6
    
    samplot = MCSamples(samples=samples,names=['I0','Re','n','x0','y0','varepsilon','PA'],labels=['I_0 [e-]', 'R_e [px]','n','x_0 [px]','y_0 [px]',r'\varepsilon','PA [º]'])
    g.triangle_plot([samplot],
        filled=True,
        #legend_labels=[r'$I_0$=0.3, $R_e$=100, n=0.5, $x_0$=120, $y_0$=110, $\varepsilon$=0.4, PA=30'], 
        line_args=[{'ls':'--', 'color':'green'},
                    {'lw':2, 'color':'darkblue'}], 
        contour_colors=['darkblue'],
        title_limit=1, # first title limit (for 1D plots) is 68% by default
        markers={'I0':mn[0].item(),'Re':mn[1].item(),'n':mn[2].item(),'x0':mn[3].item(),'y0':mn[4].item(),'varepsilon':mn[5].item(),'PA':mn[6].item()})
    for i in range(0,len(g.subplots)):
        g.subplots[len(g.subplots)-1][i].xaxis.set_minor_locator(AutoMinorLocator());g.subplots[i][0].yaxis.set_minor_locator(AutoMinorLocator())
    plt.savefig(path+"/triangular_"+ name[:-5], bbox_inches='tight', pad_inches=0.02)  
    plt.close()
    return

def trianguar_plots_corner(mn,samples,path,name):
    plt.rc('xtick', labelsize=14)    
    plt.rc('ytick', labelsize=14) 
    
    fig=corner.corner(samples,var_names=['I0','Re','n','x0','y0','varepsilon','PA'],labels=[r'$I_0$ [e-]', r'$R_e$ [px]',r'$n$',r'$x_0$ [px]',r'$y_0$ [px]',r'$\varepsilon$',r'$PA$ [º]'],
    show_titles=True,max_n_ticks=3, title_kwargs={"fontsize":14},label_kwargs=dict(fontsize=14),plot_datapoints=True,
    fill_contours=True,use_math_text=True)
    fig.subplots_adjust(right=1.5,top=1.5)
    corner.overplot_lines(fig, mn.detach(), color="C2")
    corner.overplot_points(fig, mn.detach()[None], marker="o", color="C2")
    for ax in fig.axes:
        plt.tick_params(axis="x", direction="in", length=7, width=1.2, color="k")
        plt.tick_params(axis="y", direction="in", length=7, width=1.2, color="k")
    
        ax.xaxis.set_tick_params(rotation=35)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction="in",which='minor', length=4, color='k')
    fig.savefig(path+"/triangular_"+ name[:-5],dpi=300,pad_inches=0.04,bbox_inches='tight')
    plt.close()
    return
def iter_gal_parallel(gal_files,frame_names,psf_values,objID,run,camcol,field,filte,length_run,length_field,mask_files,index,width_moff,power_moff,gain,sky_val,readout_noise):
# gal_lim = np.arange(0,int(1),1)

# for index in gal_lim:
    if comm_params:
        data=fits.getdata(gal_dir+'/'+gal_files[int(index)])*gain
    if not comm_params:
        index_2 = np.where((objID ==int(gal_files[int(index)][:-5])))[0][0]
        count_run = length_run-len(str(run[index_2]))
        run_new=count_run*'0'+str(run[index_2])
        count_field = length_field-len(str(field[index_2]))
        field_new=count_field*'0'+str(field[index_2])
        frame_name = "frame-%s"%filte +"-%s"%run_new+"-%s"%(str(camcol[index_2]))+"-%s"%field_new+".fits"
        if filte=="r":ind_filt=0
        if filte=="i":ind_filt=1
        if filte=="u":ind_filt=2
        if filte=="z":ind_filt=3
        if filte=="g":ind_filt=4
        
        index_3 = np.where((frame_names==frame_name))[0][0]
        width_moff=psf_values[index_3][0]
        power_moff=psf_values[index_3][1]
        
        phot_field_data = fits.getdata(file_params+"/"+"photoField"+"-%s"%run_new+"-%s"%(str(camcol[index_2]))+".fits")
        gain = np.mean(phot_field_data["Gain"][:,ind_filt])
        sky_val = np.mean(phot_field_data["Sky"][:,ind_filt])
        readout_noise = np.mean(phot_field_data["Dark_variance"][:,ind_filt])
        nmgy_percount = np.mean(phot_field_data["NMGYPERCOUNT"][:,ind_filt])
        data=fits.getdata(gal_dir+'/'+gal_files[int(index)])*gain/nmgy_percount
    if mask_files is None:
        mask = np.zeros((data.shape[0],data.shape[1]))
    if mask_files is not None:
        mask = fits.getdata(mask_dir+'/'+ mask_files[index])

    #box = data[(data.shape[0]//2)-10:(data.shape[0]//2)+10,(data.shape[1]//2)-10:(data.shape[1]//2)+10]
    norm_I = 1
    #data = data/norm_I
    
    # Noise standard deviation map in counts
    noise_map = np.sqrt(np.abs(data) + (sky_val * gain) + readout_noise**2)
   
    # =============================================================================
    # Generation of the PSF
    # =============================================================================
    ny, nx = data.shape
    moff_x = np.floor(nx / 2.0)
    moff_y = np.floor(ny / 2.0)
    
    psf_class = psf(xsize,ysize,psf_imgname,gauss_amp,mean_x, mean_y, theta_rot,\
                    stdv_x,stdv_y,moff_amp,xsize//2,ysize//2,width_moff,power_moff)
    class_method = getattr(psf_class, psf_type)
    psf_image = class_method() 
    
    # =============================================================================
    # Application of the bayesian method
    # =============================================================================
    
    hamiltonian_class = hamiltonian_model(data.astype(np.float64),psf_image,mask,sky_sigm,readout_noise,gain,norm_I, noise_map)
    samples = hamiltonian_class.hamiltonian_sersic()   
    return(samples,hamiltonian_class,data,gal_files[int(index)])
# =============================================================================
# Creation of directories
# =============================================================================
if not os.path.isdir(dir_img_final): os.makedirs(dir_img_final)
if not os.path.isdir(dir_img_final+"/Images"): os.makedirs(dir_img_final+"/Images")
if not os.path.isdir(dir_img_final+"/Log_Images"): os.makedirs(dir_img_final+"/Log_Images")
if not os.path.isdir(dir_img_final+"/Triangular_plots"): os.makedirs(dir_img_final+"/Triangular_plots")
if use_simu: 
    if not os.path.isdir(dir_img_final+"/Input_plots"): os.makedirs(dir_img_final+"/Input_plots")
# =============================================================================
# Choosing color map and reading data
# =============================================================================
cmap = plt.get_cmap('cmr.torch')

use_simu = str(use_simu).lower() in ['yes']
comm_params = str(comm_params).lower() in ['yes']
if use_simu:
    galaxies,Ie,Re,n, ba_b, PA_bulge, B_T,Mag_tot= data_reader(user_in_file)

gal_files = [ f for f in os.listdir(gal_dir) if f.endswith(".fits")]
gal_files.sort()

if use_simu:
    mask_files=None
if not use_simu:
    mask_files = [ f for f in os.listdir(mask_dir) if f.endswith(".fits")]
    mask_files.sort()

if  comm_params:
    frame_names=None
    psf_values=None
if not comm_params:
    frame_names=np.loadtxt(params_dir,skiprows=1,usecols=(0),dtype=str)
    psf_values=np.loadtxt(params_dir,skiprows=1,usecols=(1,2))
    

csv_file = pd.read_csv(gal_csv,header=1)
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
n_jobs=5
gal_lim = np.arange(0,len(gal_files),1)#prefer="threads"
gal_lim_list = np.array_split(gal_lim,len(gal_files)//n_jobs)
results_list = []
for i in range(0,len(gal_lim_list)):
    index_array = gal_lim_list[i]
    result = Parallel(n_jobs=n_jobs,prefer="threads")(delayed(iter_gal_parallel)
                            (gal_files,frame_names,psf_values,objID,run,camcol,field,filte,length_run,length_field,mask_files,index,width_moff,power_moff,gain,sky_val,readout_noise)
                            for index in index_array)
# =============================================================================
# Make all plots
# =============================================================================
    for j in range(0,len(index_array)):
        mn = torch.tensor(np.median(result[j][0], axis=0))
        results_list.append(result[j][0])
        compare_plots(mn,result[j][1],result[j][0],result[j][2],{"x":"x","y":"y"},\
                      {"Data":"Data","Model":"Model","Residual map": "Residual map"},\
                          r"I [$e^{-}$]",dir_img_final+"/Images",result[j][3],False)
        compare_plots(mn,result[j][1],result[j][0],result[j][2],{"x":"x","y":"y"},\
                      {"Data":"Data","Model":"Model","Residual map": "Residual map"},\
                          r"$\mu [mag/arcsec^2]$",dir_img_final+"/Log_Images",result[j][3],True)
        trianguar_plots_corner(mn,result[j][0],dir_img_final+"/Triangular_plots",result[j][3])
# =============================================================================
# Save results of each parameter
# =============================================================================
mn = [(np.median(results_list[f], axis=0)) for f in range(0,len(results_list))]
final_list = [[f] for f in zip(mn)]
final_list_1=np.concatenate( final_list,axis=0)
final_list_2=np.concatenate( final_list_1,axis=0)
df = pd.DataFrame(final_list_2, columns = ['#I0', 'Re','n','x0','y0','e','theta'])
df.to_csv(dir_img_final+'/test.txt', index = False, sep=' ')
# =============================================================================
# Compare results with the input file (solo para las simulaciones)
# Esto seguramente lo quite, asi que aqui va un poco de hard code
# =============================================================================
if use_simu:
    def plotito(x,y,name):
        fig, ax = plt.subplots(figsize=(15, 15))
        r = np.corrcoef(x, y)[0,1]
        poly_deg = 1 
        polynomial_fit_coeff = np.polyfit(x, y, poly_deg)
        lon_intrp_2 = np.polyval(polynomial_fit_coeff, x)
        plt.plot(x, lon_intrp_2, 'r-',label='Correlation'+'_'+name+ ' = ' + "{:.9f}".format(r))
        plt.plot( x,y, "k.")
        plt.tick_params(axis="x", direction="in", length=7, width=1.2, color="k")
        plt.tick_params(axis="y", direction="in", length=7, width=1.2, color="k")
        ax.set_ylabel("Input", fontsize = 16)
        ax.set_xlabel("Output", fontsize =16)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction="in",which='minor', length=4, color='k')
        plt.legend(loc="upper left")
        plt.savefig(dir_img_final+"/Input_plots"+"/inputoutput_"+ name, bbox_inches='tight', pad_inches=0.02)  
    
        plt.close()
        return ()
    Ie_sim,Re_sim,n_sim,x_sim,y_sim,e_sim,theta_sim = final_list_2.T
    Ie_input,Re_input,n_input, e_input, theta_input= Ie[gal_lim],Re[gal_lim],n[gal_lim], 1-ba_b[gal_lim], PA_bulge[gal_lim]
    
    #Ie
    plotito(Ie_sim,Ie_input, "Ie")
    #Re
    
    
    plotito(Re_sim,Re_input, "Re")
    
    
    #n
    
    
    plotito(n_sim,n_input, "n")
    
    
    #e
    plotito(e_sim,e_input, "e")
    
    
    #Pa
    
    plotito(theta_sim,theta_input, "theta")

# Calculate the elapsed time
elapsed_time = time() - start_time
print("Elapsed time: %0.10f seconds." % elapsed_time)
