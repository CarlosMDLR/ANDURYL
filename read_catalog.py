#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:24:04 2023

@author: cmr
"""
import pandas as pd
import os
import numpy as np
from astropy.io import fits

csv_file_path= "./SDSS_catalog/fits_data/catalogo_SDSS_run.csv"
csv_file = pd.read_csv(csv_file_path,header=1)
data_radius =fits.getdata('./SDSS_catalog/A2142_memberLiu_sigma.fits')

objID=csv_file["objID" ]
run=(csv_file["run" ])
camcol=(csv_file["camcol" ])
field=(csv_file["field" ])
rowc_i=(csv_file["rowc_i" ])
colc_i=(csv_file["colc_i" ])

filte="i"

galaxies_path = "./SDSS_catalog/gal_imgs/"
fits_path= "./SDSS_catalog/fits_data/"

if not os.path.isdir(galaxies_path): os.makedirs(galaxies_path)
length_run= 6
length_field=4
for i in range(0, len(objID)):
    count_run = length_run-len(str(run[i]))
    run_new=count_run*'0'+str(run[i])
    count_field = length_field-len(str(field[i]))
    field_new=count_field*'0'+str(field[i])
    frame_name = "frame-%s"%filte +"-%s"%run_new+"-%s"%(str(camcol[i]))+"-%s"%field_new+".fits"
    
    fits_file= fits_path+frame_name
    
    rowc_i_new= int(np.round(rowc_i[i]))
    colc_i_new= int(np.round(colc_i[i]))
    
    data=fits.getdata(fits_file)
    index = np.where((data_radius["SDSS"]==objID[i]))[0][0]
    
    radius = np.round(10*data_radius["petroR90_r"][index])
    if rowc_i_new-int(radius)<0 or colc_i_new-int(radius)<0:
        radius = np.round(3*data_radius["petroR90_r"][index])
        new_data = data[rowc_i_new-int(radius):rowc_i_new+int(radius), colc_i_new-int(radius):colc_i_new+int(radius)]
    else:
        new_data = data[rowc_i_new-int(radius):rowc_i_new+int(radius), colc_i_new-int(radius):colc_i_new+int(radius)]
    hdu = fits.PrimaryHDU(new_data)
    hdu.writeto(galaxies_path+str(objID[i])+".fits",overwrite=True)