"""
This script takes an input stack and divide it in tiles. This is done to be later used with TeraStitcher for
testing the Steps 3/4/5 independently.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import numpy as np
from skimage.io import imread, imsave
import os
from pathlib import Path
import pyapr

# Parameters
file_path = r'/media/sf_shared_folder_virtualbox/PV_interneurons/PV_interneurons.tif'
output_apr = r'/media/sf_shared_folder_virtualbox/multitile_registration/apr'
output_tiff = r'/media/sf_shared_folder_virtualbox/multitile_registration/tiff'
dH = 4
dV = 4
overlap_H = 25
overlap_V = 25
tiletype = 'tiff3D'

# Load data
data = imread(file_path)
dv = int(data.shape[1]*(1-overlap_V/100)/dV)
dh = int(data.shape[2]*(1-overlap_H/100)/dH)

def get_coordinates(v, dV, h, dH):
    x = int(v*dV)
    y = int(h*dH)
    x_noise = int(max(v*dV + np.random.randn(1)*5, 0))
    y_noise = int(max(h*dH + np.random.randn(1)*5, 0))
    return (x, y, x_noise, y_noise)

# Save data as separate tiles
noise_coordinates = np.zeros((dH*dV, 4))
for v in range(dV):
    for h in range(dH):
        (x, y, x_noise, y_noise) = get_coordinates(v, dv, h, dh)
        if h == 0:
            row_folder_tiff = os.path.join(output_tiff, '{:06d}'.format(10*x))
            Path(row_folder_tiff).mkdir(parents=True, exist_ok=True)
            row_folder_apr = os.path.join(output_apr, '{:06d}'.format(10*x))
            Path(row_folder_apr).mkdir(parents=True, exist_ok=True)

        depth_folder_tiff = os.path.join(row_folder_tiff, '{:06d}_{:06d}'.format(10*x, 10*y))
        Path(depth_folder_tiff).mkdir(parents=True, exist_ok=True)
        noise_coordinates[v*dH+h, :] = [x_noise, y_noise, x_noise+int(data.shape[1]/dV), y_noise+int(data.shape[1]/dV)]
        if tiletype == 'tiff2D':
            for i in range(data.shape[0]):
                imsave(os.path.join(depth_folder_tiff, '{:06d}.tif'.format(10*i)),
                       data[i, x_noise:x_noise+int(data.shape[1]/dV), y_noise:y_noise+int(data.shape[1]/dV)],
                       check_contrast=False)
        elif tiletype == 'tiff3D':
            imsave(os.path.join(depth_folder_tiff, '000000.tif'),
                   data[:, x_noise:x_noise + int(data.shape[1] / dV), y_noise:y_noise + int(data.shape[1] / dV)],
                   check_contrast=False)
        else:
            raise ValueError('Error: unknown tiletype.')

        # Parameters for PV_interneurons
        par = pyapr.APRParameters()
        par.auto_parameters = False  # really heuristic and not working
        par.sigma_th = 562.0
        par.grad_th = 49.0
        par.Ip_th = 903.0
        par.rel_error = 0.2
        par.gradient_smoothing = 2.0
        par.dx = 1
        par.dy = 1
        par.dz = 3

        # Parameters for Nissl on 2P mouse
        # par.auto_parameters = False  # really heuristic and not working
        # par.sigma_th = 18162.0
        # par.grad_th = 3405.0
        # par.Ip_th = 5351.0
        # par.rel_error = 0.2
        # par.gradient_smoothing = 1.0
        # par.dx = 1
        # par.dy = 1
        # par.dz = 1

        # Convert data to APR
        apr, parts = pyapr.converter.get_apr(
            image=data[:, x_noise:x_noise + int(data.shape[1] / dV), y_noise:y_noise + int(data.shape[1] / dV)]
            , params=par, verbose=False)
        depth_folder_apr = os.path.join(row_folder_apr, '{:06d}_{:06d}'.format(10*x, 10*y))
        Path(depth_folder_apr).mkdir(parents=True, exist_ok=True)
        pyapr.io.write(os.path.join(depth_folder_apr, '000000.apr'), apr, parts)

# Save coordinates
np.savetxt(os.path.join(output_apr, 'real_displacements.csv'), noise_coordinates, fmt='%1.d', delimiter=',')
np.savetxt(os.path.join(output_tiff, 'real_displacements.csv'), noise_coordinates, fmt='%1.d', delimiter=',')

# Run Terastitcher from python
# Set cwd to terastitcher exe folder
# terastitcher_path = r'C:\Program Files\TeraStitcher-Qt5-standalone 1.10.18\bin'
# data_path = r'C:\Users\jscholler\Desktop\shared_folder_virtualbox\multitile_registration\tiff'
# os.chdir(terastitcher_path)
# from time import time
# t = time()
# # Do step 1 (load data and create XML)
# str_step1 = r'terastitcher --import --volin="{}" --ref1=y --ref2=x --ref3=z --vxl1=1 --vxl2=1 --vxl3=1 --volin_plugin="TiledXY|3Dseries"'.format(data_path)
# os.system(str_step1)
# # Do step 2 (align)
# str_step2 = r'terastitcher --displcompute --projin="{}\xml_import.xml" --subvoldim=268'.format(data_path)
# os.system(str_step2)
# # Do step 3 (project)
# str_step3 = r'terastitcher --displproj --projin="{}\xml_displcomp.xml"'.format(data_path)
# os.system(str_step3)
# # Do step 4 (threshold)
# str_step4 = r'terastitcher --displthres --projin="{}\xml_displproj.xml" --threshold=0.7'.format(data_path)
# os.system(str_step4)
# # Do step 5 (place)
# str_step5 = r'terastitcher --placetiles --projin="{}\xml_displthres.xml"'.format(data_path)
# os.system(str_step5)
# print('\n\nElapsed time: {} s.'.format(time()-t))