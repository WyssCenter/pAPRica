"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
from time import time
import napari
import pandas as pd
import os
from tqdm import tqdm
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import numpy as np
import matplotlib.pyplot as plt
import pyapr

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

    pyapr.numerics.gradient(apr, parts, dx, dimension=1, delta=par.dx, sobel=sobel)
    pyapr.numerics.gradient(apr, parts, dy, dimension=0, delta=par.dy, sobel=sobel)

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
    pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=5*5*5)
    # pyapr.numerics.transform.remove_large_objects(apr, cc, max_volume=128)

    return cc


# Convert data to APR
# path = '/media/hbm/HDD_data/HOLT_011398_LOC000_20210721_170347/VW0'
# for c in [2]:
#     tiles = pipapr.parser.tileParser(path, frame_size=2048, ncol=7, nrow=7, ftype='tiff2D', channel=c)
#     converter = pipapr.converter.tileConverter(tiles)
#     converter.batch_convert_to_apr(Ip_th=105, rel_error=0.4, gradient_smoothing=10, path='/media/hbm/SSD1/COLM/Holmaat/chlll' + str(c))

# # Stitch data
tiles1 = pipapr.parser.tileParser(path='/media/hbm/SSD1/COLM/Holmaat/ch1', frame_size=2048, ftype='apr')
stitcher1 = pipapr.stitcher.tileStitcher(tiles1, overlap_h=29.402, overlap_v=22.175)
stitcher1.set_overlap_margin(5)
# stitcher1.compute_registration_fast()
# stitcher1.compute_registration_from_max_projs()
# stitcher1.save_database(os.path.join(path, 'registration_results.csv'))
stitcher1.database = pd.read_csv(os.path.join('/media/hbm/SSD1/COLM/Holmaat/ch1/registration_results.csv'))

# Compare segmentation with theoretical one from motor position
stitcher_th = pipapr.stitcher.tileStitcher(tiles1, overlap_h=29.402, overlap_v=22.175)
stitcher_th.compute_expected_registration()
pipapr.viewer.compare_stitching(loc=None, dim=0, n_proj=10, stitcher1=stitcher1, stitcher2=stitcher_th, color=True)

# Apply stitching to other tiles
tiles0 = pipapr.parser.tileParser(path='/media/hbm/SSD1/COLM/Holmaat/ch0', frame_size=2048, ftype='apr')
stitcher01 = pipapr.stitcher.tileStitcher(tiles0, overlap_h=29.402, overlap_v=22.175)
stitcher01.database = stitcher1.database
stitcher0 = pipapr.stitcher.channelStitcher(stitcher1, ref=tiles1, moving=tiles0)
stitcher0.set_regularization(reg_y=20, reg_x=20, reg_z=20)
# stitcher0.compute_rigid_registration()
# pipapr.viewer.compare_stitching(stitcher1=stitcher0, stitcher2=stitcher01, loc=None, dim=0, n_proj=0)

# Atlas data
# tiles2 = pipapr.parser.tileParser(path='/media/hbm/SSD1/COLM/Holmaat/chlll2', frame_size=2048, ftype='apr')
tiles2 = pipapr.parser.tileParser('/media/hbm/HDD_data/HOLT_011398_LOC000_20210721_170347/VW0', frame_size=2048, ncol=7, nrow=7, ftype='tiff2D', channel=2)
merger = pipapr.stitcher.tileMerger(tiles2, stitcher1.database)
merger.set_downsample(8)
merger.merge_max()
merger.merged_data[merger.merged_data>500] = 500
merger.merged_data[merger.merged_data<110] = 110
atlaser = pipapr.atlaser.tileAtlaser.from_merger(merger=merger, original_pixel_size=[5, 1.42, 1.42])
par = {'affine-n-steps': 7,
       'affine-use-n-steps': 9}
atlaser.register_to_atlas(output_dir='/media/hbm/SSD1/COLM/Holmaat/atlas3', orientation='sal', debug=True, params=par)


# Segment dataset
trainer = pipapr.segmenter.tileTrainer(tiles1[6+2],
                                       func_to_compute_features=compute_features,
                                       func_to_get_cc=get_cc_from_features)
trainer.manually_annotate()
trainer.train_classifier()
trainer.segment_training_tile(bg_label=2)

segmenter = pipapr.segmenter.tileSegmenter.from_trainer(trainer)
for tile in tiles1:
    segmenter.compute_segmentation(tile)