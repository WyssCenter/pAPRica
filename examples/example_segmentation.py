"""
This is a script that shows how to segment data.


By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
import pyapr
import numpy as np
from time import time

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

    pyapr.filter.gradient(apr, parts, dz, dim=2, delta=par.dz)
    pyapr.filter.gradient(apr, parts, dx, dim=1, delta=par.dx)
    pyapr.filter.gradient(apr, parts, dy, dim=0, delta=par.dy)
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
    pyapr.filter.gradient(apr, dz, dz2, dim=2, delta=par.dz)
    pyapr.filter.gradient(apr, dx, dx2, dim=1, delta=par.dx)
    pyapr.filter.gradient(apr, dy, dy2, dim=0, delta=par.dy)
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
    gradmag = pyapr.filter.gradient_magnitude(apr, parts, deltas=(par.dz, par.dx, par.dy))
    return gradmag


def compute_inplane_gradmag(apr, parts, sobel=True):
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
    dx = pyapr.FloatParticles()
    dy = pyapr.FloatParticles()

    pyapr.filter.gradient(apr, parts, dx, dim=1, delta=par.dx)
    pyapr.filter.gradient(apr, parts, dy, dim=0, delta=par.dy)

    dx = np.array(dx, copy=False)
    dy = np.array(dy, copy=False)

    return pyapr.FloatParticles(np.sqrt(dx**2 + dy**2))


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
    output = pyapr.FloatParticles()
    pyapr.filter.convolve(apr, parts, stencil, output)
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

# First we define the path where the data is located
path = '/home/user/folder_containing_data'

# If you don't have any data to try on, you can run the 'example_create_synthetic_dataset.py' script

# We then parse this data using the parser
tiles = pipapr.tileParser(path=path, frame_size=2048, ftype='apr')

# Here we will manually train the segmenter on the first tile
trainer = pipapr.tileTrainer(tiles[0, 0], func_to_compute_features=compute_features)
trainer.manually_annotate()
# We can save manual labels and add more labels later on
trainer.save_labels()
trainer.add_annotations()
# Now, let's train the classifier
trainer.train_classifier()
# We can apply it on the first tile to see how it performs
trainer.segment_training_tile()

# Now we can segment all tiles using the trained classifier
segmenter = pipapr.tileSegmenter.from_trainer(trainer)
for tile in tiles:
    segmenter.compute_segmentation(tile)

# Note that it is possible to perform the stitching and the segmentation at the same time, optimizing IO operations:
stitcher = pipapr.tileStitcher(tiles, overlap_h=20, overlap_v=20)
stitcher.activate_segmentation(segmenter)
stitcher.compute_registration()
