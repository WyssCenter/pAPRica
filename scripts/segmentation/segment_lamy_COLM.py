"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

"""
Script to process Tomas data taken on 3x2 COLM.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from time import time
import pandas as pd
import pipapr
import os
import numpy as np
import pyapr
import matplotlib.pyplot as plt


def compute_gradients(apr, parts, sobel=True):
    """
    Compute gradient for each spatial direction directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    sobel: (bool) use sobel filter to compute the gradient

    Returns
    -------
    (dx, dy, dz): (arrays) gradient for each direction
    """

    par = apr.get_parameters()
    dx = pyapr.FloatParticles()
    dy = pyapr.FloatParticles()
    dz = pyapr.FloatParticles()

    pyapr.numerics.gradient(apr, parts, dz, dimension=2, delta=par.dz, sobel=sobel)
    pyapr.numerics.gradient(apr, parts, dx, dimension=1, delta=par.dx, sobel=sobel)
    pyapr.numerics.gradient(apr, parts, dy, dimension=0, delta=par.dy, sobel=sobel)
    return dz, dx, dy


def compute_laplacian(apr, parts, grad=None, sobel=True):
    """
    Compute Laplacian for each spatial direction directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    grad: (dz, dy, dx) gradient for each direction if precomputed (faster for Laplacian computation)
    sobel: (bool) use sobel filter to compute the gradient

    Returns
    -------
    Laplacian of APR.
    """

    par = apr.get_parameters()
    if grad is None:
        dz, dx, dy = compute_gradients(apr, parts, sobel)
    else:
        dz, dx, dy = grad
    dx2 = pyapr.FloatParticles()
    dy2 = pyapr.FloatParticles()
    dz2 = pyapr.FloatParticles()
    pyapr.numerics.gradient(apr, dz, dz2, dimension=2, delta=par.dz, sobel=sobel)
    pyapr.numerics.gradient(apr, dx, dx2, dimension=1, delta=par.dx, sobel=sobel)
    pyapr.numerics.gradient(apr, dy, dy2, dimension=0, delta=par.dy, sobel=sobel)
    return dz2 + dx2 + dy2


def compute_gradmag(apr, parts, sobel=True):
    """
    Compute gradient magnitude directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    sobel: (bool) use sobel filter to compute the gradient

    Returns
    -------
    Gradient magnitude of APR.
    """

    par = apr.get_parameters()
    gradmag = pyapr.FloatParticles()
    pyapr.numerics.gradient_magnitude(apr, parts, gradmag, deltas=(par.dz, par.dx, par.dy), sobel=sobel)
    return gradmag


def gaussian_blur(apr, parts, sigma=1.5, size=11):
    """
    Compute Gaussian blur directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    sigma: (float) Gaussian blur standard deviation (kernel radius)
    size: (int) kernel size (increase with caution, complexity is not linear)

    Returns
    -------
    Blurred APR.
    """

    stencil = pyapr.numerics.get_gaussian_stencil(size, sigma, ndims=3, normalize=True)
    output = pyapr.FloatParticles()
    pyapr.numerics.filter.convolve_pencil(apr, parts, output, stencil, use_stencil_downsample=True,
                                          normalize_stencil=True, use_reflective_boundary=True)
    return output


def particle_levels(apr):
    """
    Returns apr level: for each particle the lvl is defined as the size of the particle in pixel.

    Parameters
    ----------
    apr: (APR) APR object

    Returns
    -------
    Particle level.
    """

    lvls = pyapr.ShortParticles(apr.total_number_particles())
    lvls.fill_with_levels(apr)
    lvls = np.array(lvls)

    return 2 ** (lvls.max() - lvls)


def compute_features(apr, parts):
    t = time()
    gauss = gaussian_blur(apr, parts, sigma=1.5, size=11)
    print('Gaussian computed.')

    # Compute gradient magnitude (central finite differences)
    grad = compute_gradmag(apr, gauss)
    print('Gradient magnitude computed.')
    # Compute local standard deviation around each particle
    # local_std = compute_std(apr, parts, size=5)
    # print('STD computed.')
    # Compute lvl for each particle
    lvl = particle_levels(apr)
    print('Particle level computed.')
    # Compute difference of Gaussian
    dog = gaussian_blur(apr, parts, sigma=3, size=22) - gauss
    print('DOG computed.')
    lapl_of_gaussian = compute_laplacian(apr, gauss)
    print('Laplacian of Gaussian computed.')

    print('Features computation took {} s.'.format(time()-t))

    # Aggregate filters in a feature array
    f = np.vstack((np.array(parts, copy=True),
                   lvl,
                   gauss,
                   grad,
                   lapl_of_gaussian,
                   dog
                   )).T

    return f


def get_cc_from_features(apr, parts_pred):

    # Create a mask from particle classified as cells (cell=1, background=2, membrane=3)
    parts_cells = (parts_pred == 1)

    # Use opening to separate touching cells
    pyapr.numerics.transform.opening(apr, parts_cells, binary=True, inplace=True)

    # Apply connected component
    cc = pyapr.LongParticles()
    pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)

    # # Remove small and large objects
    # pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=4)
    # pyapr.numerics.transform.remove_large_objects(apr, cc, max_volume=128)

    return cc

import re
from glob import glob
from skimage.io import imread
from tqdm import tqdm
def _load_sequence(path, channel):
    """
    Load a sequence of images in a folder and return it as a 3D array.

    Parameters
    ----------
    path: (str) path to folder where the data should be loaded.

    Returns
    -------
    v: (array) numpy array containing the data.
    """
    files = glob(os.path.join(path, '*tif'))
    n_files = len(files)

    files_sorted = list(range(n_files))
    n_max = 0
    for i, pathname in enumerate(files):
        number_search = re.search('CHN0' + str(channel) + '_PLN(\d+).tif', pathname)
        if number_search:
            n = int(number_search.group(1))
            files_sorted[n] = pathname
            if n > n_max:
                n_max = n

    files_sorted = files_sorted[:n_max]
    n_files = len(files_sorted)

    u = imread(files_sorted[0])
    v = np.empty((n_files, *u.shape), dtype='uint16')
    v[0] = u
    files_sorted.pop(0)
    for i, f in enumerate(tqdm(files_sorted)):
        v[i + 1] = imread(f)

    return v

# Parameters
path = '/media/hbm/SSD1/test_segmentation/'

# u = _load_sequence(path, 1)
# par = pyapr.APRParameters()
# par.Ip_th=350
# par.rel_error=0.2
# apr, parts = pyapr.converter.get_apr(u, params=par)
# tree_parts = pyapr.ShortParticles()
# pyapr.numerics.fill_tree_mean(apr, parts, tree_parts)
# pyapr.io.write('/media/hbm/SSD1/data.apr', apr, parts, tree_parts=tree_parts)
# pipapr.viewer.display_apr(apr, parts)


# Parse data
tiles = pipapr.parser.randomParser(path, frame_size=2048, ftype='apr')
tile = tiles[0]

# Segment tile
trainer = pipapr.segmenter.tileTrainer(tile, compute_features, get_cc_from_features)
# trainer.manually_annotate(use_sparse_labels=True)
# trainer.save_labels()
trainer.load_labels()
trainer.add_annotations()
trainer.train_classifier()
t = time()
trainer.segment_training_tile(bg_label=2, display_result=False)
print('Elapsed time: {} s.'.format(time()-t))


# trainer = pipapr.segmenter.tileTrainer(tile, func_to_compute_features=compute_features, func_to_get_cc=get_cc_from_features)
# trainer.load_labels()
# trainer.add_annotations()
# trainer.manually_annotate(use_sparse_labels=True)
# trainer.save_labels()
# trainer.train_classifier()
# trainer.segment_training_tile(bg_label=3)
# trainer.display_training_annotations()
# trainer.save_classifier()
# trainer.segment_training_tile(func_to_get_cc=get_cc_from_features)

# Apply the segmentation
# segmenter = pipapr.segmenter.tileSegmenter.from_trainer(trainer)
# segmenter.compute_segmentation(tile)

