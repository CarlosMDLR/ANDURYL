# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:59:08 2022

@author: Carlos
"""
import numpy as np 
a= np.loadtxt("setup.txt", dtype = str)
par_names = 'algo','accep_rat','like_func','max_mark', 'max_iter', 'init_guess', 'tel_scale',\
    'zero_point','psf_type', 'psf_imgname','xsize','ysize','gauss_amp','mean_x', 'mean_y', 'theta_rot','stdv_x', \
        'stdv_y','gauss_amp_2','mean_x_2', 'mean_y_2', 'theta_rot_2', 'stdv_x_2', \
        'stdv_y_2','moff_amp','moff_x', 'moff_y','width_moff','power_moff','user_in_file','syn_centr','sky_back'
d = dict(zip(par_names,a))


algo=float(d['algo'])
accep_rat=float(d['accep_rat'])
like_func=float(d['like_func'])
max_mark=float(d['max_mark'])
max_iter=float(d['max_iter'])
init_guess=float(d['init_guess'])
tel_scale=float(d['tel_scale'])
zero_point=float(d['zero_point'])
psf_type=str(d['psf_type'])
psf_imgname=str(d['psf_imgname'])
xsize=float(d['xsize'])
ysize=float(d['ysize'])
gauss_amp=float(d['gauss_amp'])
mean_x=float(d['mean_x'])
mean_y=float(d['mean_y'])
theta_rot=float(d['theta_rot'])
stdv_x=float(d['stdv_x'])
stdv_y=float(d['stdv_y'])
gauss_amp_2=float(d['gauss_amp_2'])
mean_x_2=float(d['mean_x_2'])
mean_y_2=float(d['mean_y_2'])
theta_rot_2=float(d['theta_rot_2'])
stdv_x_2=float(d['stdv_x_2'])
stdv_y_2=float(d['stdv_y_2'])
moff_amp=float(d['moff_amp'])
moff_x=float(d['moff_x'])
moff_y=float(d['moff_y'])
width_moff=float(d['width_moff'])
power_moff=float(d['power_moff'])
user_in_file=str(d['user_in_file'])
syn_centr=str(d['syn_centr'])
sky_back=str(d['sky_back'])
