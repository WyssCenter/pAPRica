"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from time import time

import pipapr
import os
import numpy as np
import pyapr


def compute_gradients(apr, parts):
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

    dz = pyapr.filter.gradient(apr, parts, dim=2, delta=par.dz)
    dx = pyapr.filter.gradient(apr, parts, dim=1, delta=par.dx)
    dy = pyapr.filter.gradient(apr, parts, dim=0, delta=par.dy)
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
    dz2 = pyapr.filter.gradient(apr, parts, dim=2, delta=par.dz)
    dx2 = pyapr.filter.gradient(apr, parts, dim=1, delta=par.dx)
    dy2 = pyapr.filter.gradient(apr, parts, dim=0, delta=par.dy)
    return dz2 + dx2 + dy2


def compute_gradmag(apr, parts):
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
    gradmag = pyapr.filter.gradient_magnitude(apr, parts, deltas=(par.dz, par.dx, par.dy))
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

    stencil = pyapr.filter.get_gaussian_stencil(size, sigma, ndims=3, normalize=True)
    return pyapr.filter.convolve(apr, parts, stencil)


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
    gauss = gaussian_blur(apr, parts, sigma=1.5, size=11)
    print('Gaussian computed.')

    # Compute gradient magnitude (central finite differences)
    grad = compute_gradmag(apr, gauss)
    print('Gradient magnitude computed.')
    # Compute lvl for each particle
    lvl = particle_levels(apr)
    print('Particle level computed.')
    # Compute difference of Gaussian
    dog = gaussian_blur(apr, parts, sigma=3, size=22) - gauss
    print('DOG computed.')
    lapl_of_gaussian = compute_laplacian(apr, gauss)
    print('Laplacian of Gaussian computed.')

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

    # Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
    parts_cells = (parts_pred == 1)

    # Use opening to separate touching cells
    pyapr.morphology.opening(apr, parts_cells, radius=1, binary=True, inplace=True)

    # Apply connected component
    cc = pyapr.LongParticles()
    pyapr.measure.connected_component(apr, parts_cells, cc)

    # Remove small objects
    # cc = pyapr.numerics.transform.remove_small_objects(apr, cc, 128)

    return cc


# Parameters
path = r'./data/apr/'
path_classifier=r'../data/random_forest_n100.joblib'
n = 1

# Parse data
t_ini = time()
tiles = pipapr.parser.tileParser(path, frame_size=512, ftype='apr')
t = time()

# Stitch
stitcher = pipapr.stitcher.tileStitcher(tiles, overlap_h=25, overlap_v=25)

t = time()
stitcher.compute_registration()
print('Elapsed time old registration: {} s.'.format((time()-t)/n))
t = time()
stitcher.compute_registration_fast()
print('Elapsed time new registration on RAM: {} s.'.format((time()-t)/n))
t = time()
stitcher.compute_registration_fast(on_disk=True)
print('Elapsed time new registration on disk: {} s.'.format((time()-t)/n))

stitcher.save_database(os.path.join(path, 'registration_results.csv'))

# Segment and extract objects across the whole volume
trainer = pipapr.tileTrainer(tiles[0],
                             func_to_compute_features=compute_features,
                             func_to_get_cc=get_cc_from_features)
trainer.manually_annotate()
trainer.train_classifier(n_estimators=100)
segmenter = pipapr.segmenter.multitileSegmenter(tiles, stitcher.database, clf=trainer.clf,
                                                func_to_compute_features=compute_features,
                                                func_to_get_cc=get_cc_from_features)
segmenter.compute_multitile_segmentation(save_cc=True)

# Display result
viewer = pipapr.viewer.tileViewer(tiles, stitcher.database, segmentation=True, cells=segmenter.cells)
viewer.display_all_tiles(pyramidal=True, downsample=1, contrast_limits=[0, 3000])