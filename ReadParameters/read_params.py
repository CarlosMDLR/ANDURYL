# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:59:08 2022

@author: Carlos
"""
import numpy as np 

def data_reader(txtname):
    galaxy= np.loadtxt(txtname, dtype="str",skiprows=1,ndmin=1,usecols=0)
    Ie,Re,n, ba_b, PA_bulge, B_T, X_center, Y_center, chi_2 = np.loadtxt(txtname, unpack=True,skiprows=1,ndmin=2,usecols=(1,2,3,4,5,6,7,8,9))
    return(galaxy,Ie,Re,n, ba_b, PA_bulge, B_T, X_center, Y_center, chi_2)

a= np.loadtxt("setup.txt", dtype = str)
par_names = 'algo','accep_rat','like_func','max_mark', 'max_iter', 'init_guess', 'tel_scale',\
    'zero_point','psf_type', 'psf_imgname','xsize','ysize','conv_method','seing_pxl','sky_val',\
        'inc_pois_nois','gain','readout_noise','in_exp_time','exp_time','exp_map','bcgr_nois',\
            'sky_sigm', 'weight_map','log_img','gauss_amp','mean_x', 'mean_y', 'theta_rot','stdv_x',\
                'stdv_y','gauss_amp_2','mean_x_2', 'mean_y_2', 'theta_rot_2', 'stdv_x_2',\
                    'stdv_y_2','moff_amp','moff_x', 'moff_y','width_moff','power_moff','user_in_file','syn_centr','sky_back'
d = dict(zip(par_names,a))

#==============================================================================
# GENERAL MCMC INPUTS
#==============================================================================
algo=float(d['algo'])
accep_rat=float(d['accep_rat'])
like_func=float(d['like_func'])
max_mark=float(d['max_mark'])
max_iter=float(d['max_iter'])
init_guess=float(d['init_guess'])
#==============================================================================
# IMAGE/INSTRUMENT INFORMATION
#==============================================================================
tel_scale=float(d['tel_scale'])
zero_point=float(d['zero_point'])
psf_type=str(d['psf_type'])
psf_imgname=str(d['psf_imgname'])
xsize=float(d['xsize'])
ysize=float(d['ysize'])
conv_method=float(d['conv_method'])
seing_pxl=float(d['seing_pxl'])
sky_val=float(d['sky_val'])
inc_pois_nois=float(d['inc_pois_nois'])
gain=float(d['gain'])
readout_noise=float(d['readout_noise'])
in_exp_time=float(d['in_exp_time'])
exp_time=float(d['exp_time'])
exp_map=str(d['exp_map'])
bcgr_nois=str(d['bcgr_nois'])
sky_sigm=float(d['sky_sigm'])
weight_map=str(d['weight_map'])
log_img=float(d['log_img'])
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
#==============================================================================
# COMPONENTS TO FIT		   
#==============================================================================
user_in_file=str(d['user_in_file'])
syn_centr=str(d['syn_centr'])
sky_back=str(d['sky_back'])


galaxies,Ie,Re,n, ba_b, PA_bulge, B_T, X_center, Y_center, chi_2= data_reader(user_in_file)