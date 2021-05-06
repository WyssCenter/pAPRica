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
file_path = r'/mnt/Data/Interneurons/PV_interneurons.tif'
output_folder = r'/mnt/Data/Interneurons/apr'
dH = 4
dV = 4
overlap_H = 25
overlap_V = 25
output_format = 'apr'

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
        noise_coordinates[v*dH+h, :] = [x_noise, y_noise, x_noise+int(data.shape[1]/dV), y_noise+int(data.shape[1]/dV)]

        if output_format == 'tiff2D':
            folder_sequence = os.path.join(output_folder, '{}_{}'.format(v, h))
            Path(folder_sequence).mkdir(parents=True, exist_ok=True)
            for i in range(data.shape[0]):
                imsave(os.path.join(folder_sequence, '{:06d}.tif'.format(10*i)),
                       data[i, x_noise:x_noise+int(data.shape[1]/dV), y_noise:y_noise+int(data.shape[1]/dV)],
                       check_contrast=False)
        elif output_format == 'tiff3D':
            imsave(os.path.join(output_folder, '{}_{}.tif'.format(v, h)),
                   data[:, x_noise:x_noise + int(data.shape[1] / dV), y_noise:y_noise + int(data.shape[1] / dV)],
                   check_contrast=False)

        elif output_format == 'apr':

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
            pyapr.io.write(os.path.join(output_folder, '{}_{}.apr'.format(v, h)), apr, parts)

        else:
            raise ValueError('Error: unknown tiletype.')


# Save coordinates
np.savetxt(os.path.join(output_folder, 'real_displacements.csv'), noise_coordinates, fmt='%1.d', delimiter=',')