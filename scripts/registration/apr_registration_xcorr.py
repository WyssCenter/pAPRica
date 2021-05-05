"""
This script demonstrate the possibility of performing registration using max-proj on APR
followed by phase cross correlation

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pyapr
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import napari
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from time import time


def display_registration(u1, u2, translation, scale=[1, 1, 1], contrast_limit=[0, 2000], opacity=0.7):
    """
    Use napari to display 2 registered volumes (works lazily with dask arrays).

    Parameters
    ----------
    u1: (array) volume 1
    u2: (array) volume 2
    translation: (array) translation vector (shape ndim) to position volume 2 with respect to volume 1
    scale: (array) scale vector (shape ndim) to display with the physical dimension if the sampling is anisotropic.
    contrast_limit: (array) contrast limit e.g. [0, 1000] to clip the displayed values and enhance the contrast.

    Returns
    -------
    None
    """

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(u1,
                         contrast_limits=contrast_limit,
                         multiscale=False,
                         name='Left',
                         scale=scale,
                         opacity=opacity,
                         colormap='red',
                         blending='additive')
        viewer.add_image(u2,
                         contrast_limits=contrast_limit,
                         multiscale=False,
                         name='Right',
                         translate=translation,
                         scale=scale,
                         opacity=opacity,
                         colormap='green',
                         blending='additive')


def get_max_proj_apr(apr, parts, plot=False):
    proj = []
    t = time()
    for d in range(3):
        # dim=0: project along Y to produce a ZX plane
        # dim=1: project along X to produce a ZY plane
        # dim=2: project along Z to produce an XY plane
        proj.append(pyapr.numerics.transform.maximum_projection(apr, parts, dim=d))
    print('\nElapsed time for APR proj: {:.3f} ms.'.format((time() - t) * 1000))

    if plot:
        fig, ax = plt.subplots(1, 3)
        for i in range(3):
            ax[i].imshow(proj[i], cmap='gray')
    return proj


def get_max_proj_pixel(apr, parts, plot=False):
    proj = []
    t = time()
    u = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts))
    for d in [2, 1, 0]:
        proj.append(np.max(u, axis=d))
    print('\nElapsed time for PIXEL proj: {:.3f} ms.'.format((time() - t) * 1000))

    if plot:
        fig, ax = plt.subplots(1, 3)
        for i in range(3):
            ax[i].imshow(proj[i], cmap='gray')
    return proj


def get_proj_shifts(proj1, proj2, upsample_factor=10):
    """
    This function computes shifts from max-projections on overlapping areas. It uses the phase cross-correlation
    to compute the shifts.

    Parameters
    ----------
    proj1: (list of arrays) max-projections for tile 1
    proj2: (list of arrays) max-projections for tile 2

    Returns
    -------
    shifts in (x, y, z) and relialability measure (0=lowest, 1=highest)
    """
    # Compute phase cross-correlation to extract shifts
    t = time()
    dzx, error_zx, _ = phase_cross_correlation(proj1[0], proj2[0],
                                               return_error=True, upsample_factor=upsample_factor)
    print('\nElapsed time for individual xcorr:\n')
    print('Proj ZX: {:.3f} ms.'.format((time()-t)*1000))
    t = time()
    dzy, error_zy, _ = phase_cross_correlation(proj1[1], proj2[1],
                                               return_error=True, upsample_factor=upsample_factor)
    print('Proj ZY: {:.3f} ms.'.format((time() - t) * 1000))
    t = time()
    dxy, error_xy, _ = phase_cross_correlation(proj1[2], proj2[2],
                                               return_error=True, upsample_factor=upsample_factor)
    print('Proj XY: {:.3f} ms.'.format((time() - t) * 1000))

    # Use the redundancy and error to optimally construct the displacement array
    dz = ((1-error_zx)*dzx[0] + (1-error_zy)*dzy[0]) / ((1-error_zx) + (1-error_zy))
    dy = ((1-error_zy)*dzy[1] + (1-error_xy)*dxy[1]) / ((1-error_zy) + (1-error_xy))
    dx = ((1-error_zx)*dzx[1] + (1-error_xy)*dxy[0]) / ((1-error_zx) + (1-error_xy))

    return np.array([dz, dx, dy]), 1-(error_xy+error_zx+error_zy)/3

# Parameters
# path = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/2P_FF0.7_AF3_Int50_Zoom3_Stacks.tif'
path = r'/media/sf_shared_folder_virtualbox/PV_interneurons/substack.tif'
mpl.use('Qt5Agg')
plt.interactive('on')
compare_with_pixel_proj = False

# Read image and divide it in two with overlap %
image_1 = imread(path)
image_1 = image_1[:, :, 800:800+512]
# image_1 = ((image_1/4)+100).astype('uint16')
# image_1 = np.tile(image_1, (4, 4, 1))
s_ini = image_1.shape

# Apply a random shift for image_2
np.random.seed(0)
random_displacement = np.random.randn(3)*[2, 20, 20]
image_2 = np.abs(np.fft.ifftn(fourier_shift(np.fft.fftn(image_1), shift=random_displacement))).astype('uint16')

# Add noise
image_2 = (image_2 + np.random.randn(*s_ini)*1000)
image_2[image_2 < 0] = 0
image_2[image_2 > 2**16-1] = 2**16-1
image_2 = image_2.astype('uint16')

# Visualize both volumes using Napari with the compensated shift
# display_registration(image_1, image_2, contrast_limit=[0, 40000],
#                      translation=[0, 0, 0])
# display_registration(image_1, image_2, contrast_limit=[0, 40000],
                     # translation=-random_displacement)

# Convert volumes to APR
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working
par.sigma_th = 26.0
par.grad_th = 3.0
par.Ip_th = 253.0
par.rel_error = 0.2
par.gradient_smoothing = 2
apr_1, parts_1 = pyapr.converter.get_apr(image_1, verbose=True, params=par)
apr_2, parts_2 = pyapr.converter.get_apr(image_2, verbose=True, params=par)

# Perform max-proj on each axis
t = time()
proj_1 = get_max_proj_apr(apr_1, parts_1, plot=False)
proj_2 = get_max_proj_apr(apr_2, parts_2, plot=False)
if compare_with_pixel_proj:
    proj_1p = get_max_proj_pixel(apr_1, parts_1, plot=False)
    proj_2p = get_max_proj_pixel(apr_2, parts_2, plot=False)
    # Verify that APR and pixel proj are the same
    for i in range(3):
        np.testing.assert_array_equal(proj_1[i], proj_1p[i])
        np.testing.assert_array_equal(proj_2[i], proj_2p[i])

# Cross-correlate each pair to extract displacement
d, relialability = get_proj_shifts(proj_1, proj_2)
print('\nElapsed time to find geometric transformation: {:.3f} ms.\n'.format((time()-t)*1000))

# Print results
accuracy = (d+random_displacement)
for i, ax in enumerate(['z', 'x', 'y']):
    print('Registration error for {} axis: {:0.3f} pixel.'.format(ax, accuracy[i]))

print('\nRelialability for registration: {:0.3f}.'.format(relialability))
