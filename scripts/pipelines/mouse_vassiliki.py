"""
Script to process the 2x2 Vassiliki data.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from time import time
import pandas as pd
from pipapr.parser import tileParser
from pipapr.stitcher import tileStitcher, tileMerger
from pipapr.viewer import tileViewer
from pipapr.atlaser import tileAtlaser
from pipapr.segmenter import tileCells
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

    # Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
    parts_cells = (parts_pred == 0)

    # Use opening to separate touching cells
    pyapr.numerics.transform.opening(apr, parts_cells, binary=True, inplace=True)

    # Apply connected component
    cc = pyapr.LongParticles()
    pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)

    # Remove small and large objects
    pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=4)
    pyapr.numerics.transform.remove_large_objects(apr, cc, max_volume=256)

    return cc

# Parameters
path_autofluo = '/mnt/Data/wholebrain/multitile/autofluo'
path_cells= '/mnt/Data/wholebrain/multitile/c1'
path_classifier = r'/mnt/Data/wholebrain/multitile/c1/random_forest_n10.joblib'
t_ini = time()

# Parse data
tiles_autofluo = tileParser(path_autofluo, frame_size=2048, overlap=868, ftype='apr')
tiles_cells = tileParser(path_cells, frame_size=2048, overlap=868, ftype='apr')

# Stitch and segment data
t = time()
stitcher = tileStitcher(tiles_autofluo)
# stitcher.activate_mask(95)
# stitcher.activate_segmentation(path_classifier, compute_features, get_cc_from_features, verbose=True)
stitcher.compute_registration()
stitcher.plot_min_trees(annotate=True)
stitcher.save_database(os.path.join(path_autofluo, 'registration_results.csv'))
print('\n\nTOTAL elapsed time for parsing, stitching and segmenting: {:.2f} s.'.format(time() - t_ini))

# Display registered tiles
# viewer = tileViewer(tiles_autofluo, stitcher.database)
# viewer.display_all_tiles(level_delta=0, contrast_limits=[0, 1000])

# Merge autofluo data
# merger = tileMerger(tiles_autofluo, stitcher.database, n_planes=2008)
# merger.set_downsample(4)
# merger.initialize_merged_array()
# merger.merge_max(mode='constant')
# merger.equalize_hist(method='opencv')
# merger.crop(background=167, ylim=[0, 733])
#
# # Display merged data
# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(merger.merged_data[250], cmap='gray')
# ax[0].set_title('YX')
# ax[1].imshow(merger.merged_data[:, 400, :], cmap='gray')
# ax[1].set_title('ZX')
# ax[2].imshow(merger.merged_data[:, :, 400], cmap='gray')
# ax[2].set_title('ZY')


# Register merged data to the atlas
pixel_size = [5, 5.26, 5.26]
# atlaser = tileAtlaser.from_merger(merger, pixel_size)
# atlasing_param = {"atlas": "allen_mouse_25um",
#                   "affine-n-steps": 6,
#                   "affine-use-n-steps": 5,
#                   "freeform-n-steps": 6,
#                   "freeform-use-n-steps": 4,
#                   "bending-energy-weight": 0.95,
#                   "grid-spacing": -10,
#                   "smoothing-sigma-reference": -1.0,
#                   "smoothing-sigma-floating": -1.0,
#                   "histogram-n-bins-floating": 128,
#                   "histogram-n-bins-reference": 128,
#                   "n-free-cpus": 4,
#                   "debug": ''}
# atlaser.register_to_atlas(output_dir='/home/jules/Desktop/test_atlasing',
#                           orientation='spr',
#                           **atlasing_param)
# Load a previously computed atlas
atlaser = tileAtlaser.from_atlas(atlas='/home/jules/Desktop/test_atlasing/atlas/registered_atlas.tiff',
                                 original_pixel_size=pixel_size,
                                 downsample=4)

# Merge cells and create a database of the full volume
cells = tileCells(tiles_cells, stitcher.database)
cells.extract_and_merge_cells()

cells_id = atlaser.get_cells_id(cells)
cells_per_regions = atlaser.get_ontology_mapping(cells_id, 0)

ncell = atlaser.get_cells_number_per_region(cells_id)
dcell = atlaser.get_cells_density_per_region(cells_id)
dcell2 = atlaser.get_cells_density(cells.cells, 2)


# Merge c1 data
merger = tileMerger(tiles_cells, stitcher.database, n_planes=2008)
merger.set_downsample(4)
merger.initialize_merged_array()
merger.merge_max(mode='constant')


from pipapr.viewer import display_heatmap
display_heatmap(dcell2, atlas=atlaser, data=merger.merged_data)