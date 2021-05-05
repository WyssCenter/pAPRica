"""
Batch convert COLM acquisition to APR and arrange the data so that it can be processed by tilemanager.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import numpy as np
import pyapr
from glob import glob
import os
from skimage.io import imread
from alive_progress import alive_bar
from time import time
import re

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

def sort_list(mylist):
    """
    Sort the folder list so that we correctly process it.
    """

    mylistsorted = list(range(len(mylist)))
    for i, pathname in enumerate(mylist):
        number_search = re.search('LOC0(\d+)', pathname[-10:])
        if number_search:
            n = int(number_search.group(1))
        else:
            raise TypeError('Couldn''t get the number')

        mylistsorted[n] = pathname
    return mylistsorted

# Parameters for batch conversion
data_path = r'/media/jules/ALICe_Ivana/LOC000_20210420_153304/VW0'
output_dir = r'/home/jules/Desktop/mouse_colm/'
n_H = 10
n_V = 10

# Parameters for APR
compress = False
par = pyapr.APRParameters()
par.rel_error = 0.2
par.gradient_smoothing = 3
par.dx = 1
par.dy = 1
par.dz = 1
par.Ip_th = 450
par.sigma_th = 95.0
par.grad_th = 15.0

# Conversion
folders = glob(os.path.join(data_path, 'LOC*'))
folders = sort_list(folders)
folders = folders[96:]
loading = []
conversion = []
writing = []
for i, f in enumerate(folders):

    t = time()
    u = load_sequence(f)
    print('Loading took {:0.2f} s.'.format(time()-t))
    loading.append(time()-t)

    t = time()
    apr, parts = pyapr.converter.get_apr(u, params=par)
    print('Conversion took {:0.2f} s.'. format(time()-t))
    conversion.append(time()-t)

    if compress:
        parts.set_compression_type(1)
        parts.set_quantization_factor(1)
        parts.set_background(180)

    t = time()

    number_search = re.search('LOC(\d+)', f[-10:])
    if number_search:
        n = int(number_search.group(1))
    else:
        raise TypeError('Couldn''t get the number')

    H = n % n_H
    V = n // n_V

    pyapr.io.write(os.path.join(output_dir, '{}_{}.apr'.format(V, H)), apr, parts, write_tree=False)
    print('Writing took {:0.2f} s.'. format(time()-t))
    writing.append(time()-t)


# # Check on a small chunk that the data is correctly parsed and aligned
# apr = []
# parts = []
# layers = []
# from viewer.pyapr_napari import display_layers, apr_to_napari_Image
# for i in [4,5,6,7]:
#     apr.append(pyapr.APR())
#     parts.append(pyapr.ShortParticles())
#     pyapr.io.read('/home/jules/Desktop/mouse_colm/multitile/2_{}.apr'.format(i), apr[-1], parts[-1])
#     position = [0, i*(2048*0.75)]
#     layers.append(apr_to_napari_Image(apr[-1], parts[-1],
#                                       mode='constant',
#                                       translate=position,
#                                       opacity=0.7,
#                                       level_delta=0))
#
# display_layers(layers)