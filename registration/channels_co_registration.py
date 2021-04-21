"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from aicsimageio import AICSImage
from skimage.io import imread
from viewer.pyapr_napari import APRArray, display_layers
import numpy as np
import os
import napari
from napari.layers import Image
import pyapr
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt


def load_czi():
    # Parameters
    file_path = r'/media/sf_shared_folder_virtualbox/210216_mouse_FGN/210216_mouse_FGN_4oC_1.czi'

    # Load data
    img_object = AICSImage(file_path)
    data = img_object.get_image_data("CZYX", S=0, T=0, M=0)

    # Display data
    with napari.gui_qt():
        viewer = napari.Viewer()
        for i, c in enumerate(['green', 'red', 'blue']):
            viewer.add_image(data[i],
                             name='Channel {}'.format(i),
                             colormap=c,
                             opacity=0.7)
    return data

def get_proj_shifts(proj1, proj2, upsample_factor=1):
    """
    This function computes shifts from max-projections on overlapping areas. It uses the phase cross-correlation
    to compute the shifts.

    Parameters
    ----------
    proj1: (list of arrays) max-projections for tile 1
    proj2: (list of arrays) max-projections for tile 2

    Returns
    -------
    shifts in (x, y, z) and error measure (0=reliable, 1=not reliable)
    """
    # Compute phase cross-correlation to extract shifts
    dzy, error_zy, _ = phase_cross_correlation(proj1[0], proj2[0],
                                               return_error=True, upsample_factor=upsample_factor)
    dzx, error_zx, _ = phase_cross_correlation(proj1[1], proj2[1],
                                               return_error=True, upsample_factor=upsample_factor)
    dyx, error_yx, _ = phase_cross_correlation(proj1[2], proj2[2],
                                               return_error=True, upsample_factor=upsample_factor)

    # Keep only the most reliable registration
    # D/z
    if error_zx < error_zy:
        dz = dzx[0]
        rz = error_zx
    else:
        dz = dzy[0]
        rz = error_zy

    # H/x
    if error_zx < error_yx:
        dx = dzx[1]
        rx = error_zx
    else:
        dx = dyx[1]
        rx = error_yx

    # V/y
    if error_yx < error_zy:
        dy = dyx[0]
        ry = error_yx
    else:
        dy = dzy[1]
        ry = error_zy

    # for i, title in enumerate(['ZY', 'ZX', 'YX']):
    #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    #     ax[0].imshow(proj1[i], cmap='gray')
    #     ax[0].set_title('dx={}, dy={}, dz={}'.format(dx, dy, dz))
    #     ax[1].imshow(proj2[i], cmap='gray')
    #     ax[1].set_title(title)
    #
    # if self.row==0 and self.col==1:
    #     print('ok')

    return np.array([dz, dy, dx]), np.array([rz, ry, rx])

def get_max_proj_apr(apr, parts, plot=False):
    """
    Get the maximum projection from 3D APR data.
    """
    proj = []
    for d in range(3):
        # dim=0: project along Y to produce a ZY plane
        # dim=1: project along X to produce a ZX plane
        # dim=2: project along Z to produce an YX plane
        proj.append(pyapr.numerics.transform.projection.maximum_projection(apr, parts, dim=d))

    if plot:
        fig, ax = plt.subplots(1, 3)
        for i, title in enumerate(['ZY', 'ZX', 'YX']):
            ax[i].imshow(proj[i], cmap='gray')
            ax[i].set_title(title)

    return proj

def gaussian_blur_apr(apr, parts, sigma=1.5, size=11):
    """
    Returns a gaussian filtered APR data.
    """
    stencil = pyapr.numerics.get_gaussian_stencil(size, sigma, 3, True)
    output = pyapr.FloatParticles()
    pyapr.numerics.filter.convolve_pencil(apr, parts, output, stencil, use_stencil_downsample=True,
                                          normalize_stencil=True, use_reflective_boundary=True)
    return output


# Parameters
path = r'/media/sf_shared_folder_virtualbox/210216_mouse_FGN/'
names = ['interneurons.tif', 'nissl.tif', 'dapi.tif']

for i, name in enumerate(names):
    if i == 0:
        tmp = imread(os.path.join(path, name))
        s = tmp.shape
        data = np.zeros((len(names), *s), dtype='uint16')
        data[i] = tmp.copy()
        del tmp
    else:
        data[i] = imread(os.path.join(path, name))

# # Display original data
# with napari.gui_qt():
#     viewer = napari.Viewer()
#     for i, c in enumerate(['green', 'red', 'blue']):
#         viewer.add_image(data[i],
#                          name='Channel {}'.format(i),
#                          colormap=c,
#                          opacity=0.7)


# # Initialize parameters for APR conversion
par = pyapr.APRParameters()
par.rel_error = 0.2              # relative error threshold
par.gradient_smoothing = 2       # b-spline smoothing parameter for gradient estimation
#                                  0 = no smoothing, higher = more smoothing
par.dx = 1.66
par.dy = 1.66                       # voxel size
par.dz = 1.5

# # Compute APR and sample particle values
# apr, parts = pyapr.converter.get_apr_interactive(data[2], params=par, verbose=True, slider_decimals=1)


# Transform to APR
aprarrays = {}
apr = {}
parts = {}
# Interneurons
par.grad_th = 2.5
par.sigma_th = 10.0
par.Ip_th = 20.0
apr['interneurons'], parts['interneurons'] = pyapr.converter.get_apr(data[0], params=par, verbose=True)
print(apr['interneurons'].computational_ratio())
aprarrays['interneurons'] = APRArray(apr['interneurons'], parts['interneurons'], type='constant')
# Nissl
par.grad_th = 6.5
par.sigma_th = 32.4
par.Ip_th = 80.0
apr['nissl'], parts['nissl'] = pyapr.converter.get_apr(data[1], params=par, verbose=True)
print(apr['nissl'].computational_ratio())
aprarrays['nissl'] = APRArray(apr['nissl'], parts['nissl'], type='constant')
# Dapi
par.grad_th = 6.5
par.sigma_th = 51.9
par.Ip_th = 60.0
apr['dapi'], parts['dapi'] = pyapr.converter.get_apr(data[2], params=par, verbose=True)
print(apr['dapi'].computational_ratio())
aprarrays['dapi'] = APRArray(apr['dapi'], parts['dapi'], type='constant')

# Add gaussian blur
parts['nissl'] = gaussian_blur_apr(apr['nissl'], parts['nissl'], sigma=10, size=20)
parts['interneurons'] = gaussian_blur_apr(apr['interneurons'], parts['interneurons'], sigma=10, size=20)

proj = {'nissl': get_max_proj_apr(apr['nissl'], parts['nissl'], plot=False),
        'dapi': get_max_proj_apr(apr['dapi'], parts['dapi'], plot=False),
        'interneurons': get_max_proj_apr(apr['interneurons'], parts['interneurons'], plot=False)}

d = {'nissl': [0, 0, 0]}
d['dapi'], err1 = get_proj_shifts(proj['nissl'], proj['dapi'], upsample_factor=10)
d['interneurons'], err2 = get_proj_shifts(proj['nissl'], proj['interneurons'], upsample_factor=10)

# Display APR in Napari using our hack
layers = []
for i, c, n in zip(range(3), ['green', 'red', 'blue'], ['interneurons', 'nissl', 'dapi']):
    layers.append(Image(data=aprarrays[n], rgb=False, multiscale=False,
                        name=n, colormap=c, opacity=0.7,
                        translate=d[n]))
display_layers(layers)