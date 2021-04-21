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
path_output = r'/media/sf_shared_folder_virtualbox/test_apr3'
image = imread(path)
rel_errors = [0.4]
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working
# 2D
# par.sigma_th = 115.0
# par.grad_th = 8.0
# par.Ip_th = 337.0
# 3D
par.sigma_th = 26.0 # Difference between dimmest object and background
# par.grad_th = 10.0
grads = [10.0, 20.0, 30.0, 50.0] #
par.Ip_th = 160.0 # Intensity threshold

for rel_error in rel_errors:
    for grad in grads:

        par.grad_th = grad
        par.rel_error = rel_error
        par.gradient_smoothing = 2
        apr, parts = pyapr.converter.get_apr(image, verbose=True, params=par)
        pyapr.io.write('temp.apr', apr, parts)
        pyapr.io.read('temp.apr', apr, parts, t=0, channel_name_apr='t', channel_name_parts='particles')
        mcr = os.path.getsize(path)/os.path.getsize('../temp.apr')

        # Compute piecewise constant reconstruction
        pc_recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)
        imsave(os.path.join(path_output, 'pc_re{}_grad{}_mcr{:0.3f}.tif'.format(rel_error, par.grad_th, mcr)), pc_recon, check_contrast=False)
        # Compute smooth reconstruction
        smooth_recon = np.array(pyapr.numerics.reconstruction.recon_smooth(apr, parts), copy=False)
        imsave(os.path.join(path_output, 'smooth_re{}_grad{}_mcr{:0.3f}.tif'.format(rel_error, par.grad_th, mcr)), smooth_recon, check_contrast=False)
        # Compute level reconstruction
        level_recon = np.array(pyapr.numerics.reconstruction.recon_level(apr), copy=False)
        imsave(os.path.join(path_output, 'level_re{}_grad{}_mcr{:0.3f}.tif'.format(rel_error, par.grad_th, mcr)), level_recon, check_contrast=False)