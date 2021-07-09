"""
Script to benchmark cell counting in synthetic dataset on a multitile dataset.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import numpy as np
import pyapr
import napari
from skimage.morphology import ball
import pyapr
import cv2 as cv
import os
from skimage.io import imsave, imread
from pathlib import Path

import pipapr.viewer

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

    # Create a mask from particle classified as cells (cell=1, background=2, membrane=3)
    parts_cells = (parts_pred == 1)

    # Use opening to separate touching cells
    # pyapr.numerics.transform.opening(apr, parts_cells, binary=True, inplace=True)

    # Apply connected component
    cc = pyapr.LongParticles()
    pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)

    # # Remove small and large objects
    # pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=1000)
    # pyapr.numerics.transform.remove_large_objects(apr, cc, max_volume=50000)

    return cc


def create_dataset(n_cells, length):
    size_max = 10
    data = np.ones([length]*3, dtype='uint16')*100
    cell_positions = (np.random.rand(n_cells, 3) * (length - size_max*2+1)).astype('uint16')

    for i in range(n_cells):
        cell_radius = int((1+np.random.rand(1)*(size_max-1)))
        cell = ball(cell_radius, dtype='uint16') * 500

        data[
        cell_positions[i, 0]: cell_positions[i, 0]+cell.shape[0],
        cell_positions[i, 1]:cell_positions[i, 1]+cell.shape[0],
        cell_positions[i, 2]:cell_positions[i, 2]+cell.shape[0]
        ] =         data[
        cell_positions[i, 0]: cell_positions[i, 0]+cell.shape[0],
        cell_positions[i, 1]:cell_positions[i, 1]+cell.shape[0],
        cell_positions[i, 2]:cell_positions[i, 2]+cell.shape[0]
        ] + cell

        cell_positions[i, :] += cell_radius

    noise = (np.random.randn(*data.shape)*np.sqrt(data)).astype('uint16')

    return data+noise, cell_positions


def match_cells(c_ref, c_pred, lowe_ratio=0.7, distance_max=3):

    if lowe_ratio < 0 or lowe_ratio > 1:
        raise ValueError('Lowe ratio is {}, expected between 0 and 1.'.format(lowe_ratio))

    # Match cells descriptors by using Flann method
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.float32(c_ref), np.float32(c_pred), k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < lowe_ratio*n.distance and m.distance < distance_max:
            good.append(m)

    ind_ref = [m.queryIdx for m in good]
    ind_pred = [m.trainIdx for m in good]

    return c_ref[ind_ref], c_pred[ind_pred]


def save_multitile(data, output_folder_apr):

    Path(output_folder_apr).mkdir(parents=True, exist_ok=True)

    dv = 512
    dh = 512
    overlap=128

    # Save data as separate tiles

    for v in range(2):
        for h in range(2):
            # Convert data to APR
            apr, parts = pyapr.converter.get_apr(
                image=data[:, v*(dv-overlap):v*(dv-overlap)+dv, h*(dh-overlap):h*(dh-overlap)+dh]
                , params=par, verbose=False)
            pyapr.io.write(os.path.join(output_folder_apr, '{}_{}.apr'.format(v, h)), apr, parts)


# Parameters
path = '/home/apr-benchmark/Desktop/data/synthetic_multitile_cells'
n_cells = 32
length = 1024-128
par = pyapr.APRParameters()
par.auto_parameters = True
par.Ip_th = 120
par.rel_error = 0.2
par.gradient_smoothing = 2

# Create dataset
data, c_ref = create_dataset(n_cells, length)
save_multitile(data, path)

# Stitch and segment
tiles = pipapr.parser.tileParser(path, frame_size=512, overlap=25)
segmenter = pipapr.segmenter.tileSegmenter.from_classifier(classifier=os.path.join(path, 'classifier.joblib'),
                                                           func_to_compute_features=compute_features,
                                                           func_to_get_cc=get_cc_from_features)
stitcher = pipapr.stitcher.tileStitcher(tiles)
stitcher.activate_segmentation(segmenter)
stitcher.compute_registration_fast()

# Find cells
cells = pipapr.segmenter.tileCells(tiles, stitcher.database)
cells.extract_and_merge_cells(lowe_ratio=0.7, distance_max=3)

# Compute reference number of cells
apr, parts = pyapr.converter.get_apr(data, params=par)
tile_ref = pipapr.loader.tileLoader(path=None, row=None, col=None, ftype='apr', neighbors_path=None, overlap=None,
                                    frame_size=1024-128, folder_root=None, neighbors=None)
tile_ref.apr = apr
tile_ref.parts = parts
segmenter.compute_segmentation(tile_ref, save_cc=False)
cells_ref = pyapr.numerics.transform.find_label_centers(tile_ref.apr, tile_ref.parts_cc)

# Display results
viewer = pipapr.viewer.tileViewer(tiles, stitcher.database, segmentation=True, cells=cells.cells)
viewer.display_all_tiles(contrast_limits=[0, 500])