# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:59:08 2022

@author: Carlos
"""
#import modules
import numpy as np 

""" 
The data from the parameter file is read, separating between when
 there are mocks generated with a Sersic profile and a Sersic+exp
"""
def data_reader(txtname):
    galaxy= np.loadtxt(txtname, dtype="str",skiprows=1,ndmin=1,usecols=0)
    Ie,Re,n, ba_b, PA_bulge, B_T, Mag_tot = np.loadtxt(txtname, unpack=True,\
                                    skiprows=1,ndmin=2,usecols=(1,2,3,4,5,6,7))
    return(galaxy,Ie,Re,n, ba_b, PA_bulge, B_T,Mag_tot)

def data_reader_bulgedisc(txtname):
    galaxy= np.loadtxt(txtname, dtype="str",skiprows=1,ndmin=1,usecols=0)
    Ie,Re,n, ba_b, PA_bulge, B_T,I0,h,q_dis,PA_disc,D_T, Mag_tot =\
    np.loadtxt(txtname, unpack=True,skiprows=1,ndmin=2,\
    usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
    return(galaxy,Ie,Re,n, ba_b, PA_bulge, B_T,I0,h,q_dis,PA_disc,D_T, Mag_tot)

a= np.loadtxt("./setup.txt", dtype = str)

par_names = 'psf_type','xsize','ysize','seing_pxl','sky_val',\
        'gain','readout_noise',\
            'sky_sigm','gauss_amp','mean_x', 'mean_y', 'theta_rot','stdv_x',\
                'stdv_y','gauss_amp_2','mean_x_2', 'mean_y_2', 'theta_rot_2',\
                    'stdv_x_2','stdv_y_2','moff_amp','moff_x', 'moff_y',\
                        'width_moff','power_moff','n_cores',\
                        'low_lim','upp_lim','to_fit','user_in_file',\
                        'gal_dir', 'mask_dir','params_dir','gal_csv',\
                        'use_simu', 'comm_params','file_params',\
                        'dir_img_final','make_plots'
d = dict(zip(par_names,a))


#==============================================================================
# IMAGE/INSTRUMENT INFORMATION
#==============================================================================

psf_type=str(d['psf_type'])
xsize=float(d['xsize'])
ysize=float(d['ysize'])
seing_pxl=float(d['seing_pxl'])
sky_val=float(d['sky_val'])
gain=float(d['gain'])
readout_noise=float(d['readout_noise'])
sky_sigm=float(d['sky_sigm'])
#==============================================================================
# GAUSSIAN PARAMETERS
#==============================================================================
gauss_amp=float(d['gauss_amp'])
mean_x=float(d['mean_x'])
mean_y=float(d['mean_y'])
theta_rot=float(d['theta_rot'])
stdv_x=float(d['stdv_x'])
stdv_y=float(d['stdv_y'])
#==============================================================================
# DOUBLE GAUSSIAN PARAMETERS 
#==============================================================================
gauss_amp_2=float(d['gauss_amp_2'])
mean_x_2=float(d['mean_x_2'])
mean_y_2=float(d['mean_y_2'])
theta_rot_2=float(d['theta_rot_2'])
stdv_x_2=float(d['stdv_x_2'])
stdv_y_2=float(d['stdv_y_2'])
#==============================================================================
# MOFFAT PARAMETERS
#==============================================================================
moff_amp=float(d['moff_amp'])
moff_x=float(d['moff_x'])
moff_y=float(d['moff_y'])
width_moff=float(d['width_moff'])
power_moff=float(d['power_moff'])

#===============================================================================================
# COMPUTING PARAMETERS		   
#===============================================================================================
n_cores = int(d['n_cores'])
low_lim = int(d['low_lim'])
upp_lim = int(d['upp_lim'])
#==============================================================================
# COMPONENTS TO FIT		   
#==============================================================================
to_fit = str(d['to_fit'])
user_in_file=str(d['user_in_file'])
gal_dir=str(d['gal_dir'])
mask_dir=str(d['mask_dir'])
params_dir=str(d['params_dir'])
gal_csv =str(d['gal_csv'])
use_simu=str(d['use_simu'])
comm_params=str(d['comm_params'])
file_params=str(d['file_params'])
dir_img_final=str(d['dir_img_final'])
make_plots=str(d['make_plots'])
