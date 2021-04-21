"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pyapr
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import os

path = r'/home/jules/Desktop/data/cells_3D_raw.tif'
path_output = r'/media/sf_shared_folder_virtualbox/test_apr_parameters'
image = imread(path)
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working

# Default parameters used when others are varying
rel_error = 0.4 # relative_error as defined in the paper
sigma_th = 26.0 # Difference between dimmest object and background
grad_th = 10.0 # Gradient threshold
Ip_th = 160.0 # Intensity threshold (gradient is set to 0 where I(y) < Ip_th)
gradient_smoothing = 2 # Lambda parameter in the BSpline smoothing


# Varying parameters:
varying_parameters = {'rel_error': [0.1, 0.2, 0.4, 0.8, 1],
                      'sigma_th': [10.0, 26.0, 50.0, 100.0, 200.0],
                      'grad_th': [2.0, 5.0, 10.0, 20.0, 50.0],
                      'Ip_th': [50.0, 160.0, 250.0, 350.0],
                      'gradient_smoothing': [0, 1, 2, 3, 4]}


for param in varying_parameters.keys():
    for value in varying_parameters[param]:

        # Set standard values
        par.rel_error = rel_error
        par.sigma_th = sigma_th
        par.grad_th = grad_th
        par.Ip_th = Ip_th
        par.gradient_smoothing = gradient_smoothing

        # Update the varying one
        setattr(par, param, value)

        apr, parts = pyapr.converter.get_apr(image, verbose=True, params=par)
        pyapr.io.write('temp.apr', apr, parts)
        pyapr.io.read('temp.apr', apr, parts, t=0, channel_name_apr='t', channel_name_parts='particles')
        mcr = os.path.getsize(path)/os.path.getsize('temp.apr')

        # Compute piecewise constant reconstruction
        pc_recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)
        imsave(os.path.join(path_output, 'pc_re{:0.1f}_sigma{}_grad{}_Ip{}_lambda{}_mcr{:0.3f}.tif'
                            .format(par.rel_error,
                                    par.sigma_th,
                                    par.grad_th,
                                    par.Ip_th,
                                    par.gradient_smoothing,
                                    mcr)), pc_recon[11], check_contrast=False)
        # Compute smooth reconstruction
        # smooth_recon = np.array(pyapr.numerics.reconstruction.recon_smooth(apr, parts), copy=False)
        # imsave(os.path.join(path_output, 'smooth_re{:0.1f}_sigma{}_grad{}_Ip{}_lambda{}_mcr{:0.3f}.tif'
        #                     .format(par.rel_error,
        #                             par.sigma_th,
        #                             par.grad_th,
        #                             par.Ip_th,
        #                             par.gradient_smoothing,
        #                             mcr)), smooth_recon[11], check_contrast=False)
        # Compute level reconstruction
        # level_recon = np.array(pyapr.numerics.reconstruction.recon_level(apr), copy=False)
        # imsave(os.path.join(path_output, 'level_re{:0.1f}_sigma{}_grad{}_Ip{}_lambda{}_mcr{:0.3f}.tif'
        #                     .format(par.rel_error,
        #                             par.sigma_th,
        #                             par.grad_th,
        #                             par.Ip_th,
        #                             par.gradient_smoothing,
        #                             mcr)), level_recon[11], check_contrast=False)