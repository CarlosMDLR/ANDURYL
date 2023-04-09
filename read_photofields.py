# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 10:03:45 2023

@author: Usuario
"""

import pandas as pd
import os
import numpy as np
from astropy.io import fits


csv_file_path= "./SDSS_catalog/fits_data/catalogo_SDSS_run.csv"
csv_file = pd.read_csv(csv_file_path,header=1)

objID=csv_file["objID" ]
run=(csv_file["run" ])
camcol=(csv_file["camcol" ])
field=(csv_file["field" ])
rowc_i=(csv_file["rowc_i" ])
colc_i=(csv_file["colc_i" ])

filte="i"

photo_path = "./SDSS_catalog/photo_field_data/"
fits_path= "./SDSS_catalog/fits_data/"

if not os.path.isdir(photo_path): os.makedirs(photo_path)
camcol_val = np.unique(camcol)
urls=np.loadtxt(fits_path+'downloads.txt', dtype=str)
urls=np.array([urls[i].replace('/frames','') for i in range(0,len(urls))])
urls=np.array([urls[i].replace('frame-g','photoField') for i in range(0,len(urls))])
urls=np.array([urls[i].replace('frame-r','photoField') for i in range(0,len(urls))])
urls=np.array([urls[i].replace('frame-i','photoField') for i in range(0,len(urls))])
urls=np.array([urls[i].replace(urls[i][-14:-9],'') for i in range(0,len(urls))])
urls=np.array([urls[i].replace('.bz2','') for i in range(0,len(urls))])
for j in camcol_val:
    urls=np.array([urls[i].replace("/"+"%s"%(j)+"/",'/') for i in range(0,len(urls))])
urls = np.unique(urls)

save = np.savetxt(photo_path+'downloads_photofield.txt',urls,fmt="%s")
#https://dr12.sdss.org/sas/dr12/boss/photoObj/301/3927/photoField-003927-2.fits