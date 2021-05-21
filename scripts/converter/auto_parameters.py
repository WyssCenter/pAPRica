"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import os
from skimage.io import imread
import numpy as np
import pyapr
from skimage import filters
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from glob import glob
import re


def plot_hist(x, nbins, ax):
    ax.hist(x.flatten(), bins=nbins, density=True)
    ax.set_yscale('log')


def plot_loghist(x, nbins, ax):
    hist, bins = np.histogram(x, bins=nbins)
    if bins[0] == 0:
        bins[0] = 1e-9
    logbins = np.logspace(np.log10(bins.min()), np.log10(bins.max()), nbins)
    ax.hist(x.flatten(), bins=logbins, density=True)
    ax.set_yscale('log')
    ax.set_xscale('log')


def load_sequence(path):
    """
    Load a sequence of images in a folder and return it as a 3D array.
    """

    files = glob(os.path.join(path, '*tif'))
    n_files = len(files)

    files_sorted = list(range(n_files))
    for i, pathname in enumerate(files):
        number_search = re.search('PLN(\d+).tif', pathname)
        if number_search:
            n = int(number_search.group(1))
        else:
            raise TypeError('Couldn''t get the number')

        files_sorted[n] = pathname

    u = imread(files_sorted[0])
    v = np.empty((n_files, *u.shape), dtype='uint16')
    v[0] = u
    files_sorted.pop(0)
    with alive_bar(n_files, force_tty=True, title='Loading sequence') as bar:
        for i, f in enumerate(files_sorted):
            v[i+1] = imread(f)
            bar()

    return v

# Set directory for 'output_steps' files
out_dir = '/home/jules/Desktop/tmp/'
fpath = '/home/jules/Desktop/data_autoparam'
files = glob(os.path.join(fpath, '*.tif'))

for file in files:
    img = imread(file).astype(np.float32)

    # Set some parameters
    par = pyapr.APRParameters()
    par.rel_error = 0.1          # relative error threshold
    par.gradient_smoothing = 3   # b-spline smoothing parameter for gradient estimation
    #                              0 = no smoothing, higher = more smoothing
    par.dx = 1
    par.dy = 1                   # voxel size
    par.dz = 1
    # threshold parameters
    par.Ip_th = 0                # regions below this intensity are regarded as background
    par.grad_th = 0              # gradients below this value are set to 0
    par.sigma_th = 0            # the local intensity scale is clipped from below to this value
    par.auto_parameters = False  # if true, threshold parameters are set automatically based on histograms

    par.output_dir = out_dir
    par.output_steps = True

    # Compute APR and sample particle values
    apr, parts = pyapr.converter.get_apr(img, params=par, verbose=True)

    grad = imread(os.path.join(out_dir, 'gradient_step.tif'))
    lis = imread(os.path.join(out_dir, 'local_intensity_scale_step.tif'))

    fig, ax = plt.subplots(1, 3)
    plot_hist(np.log10(img+1), 1000, ax[0])
    ax[0].set_title('I - ' + os.path.basename(file))
    plot_loghist(grad, 100, ax[1])
    ax[1].set_title('Grad')
    plot_hist(np.log10(lis+1), 1000, ax[2])
    ax[2].set_title('LIS')