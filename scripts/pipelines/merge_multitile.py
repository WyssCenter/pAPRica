"""
This script downsample the data with a power of 2 using APR and then interpolate linearly the data on a given resolution
in order to run the atlasing pipapr (brainreg).

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from pipapr.stitcher import tileMerger
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator
import napari
from skimage.io import imsave

# Parameters
path = r'/mnt/Data/wholebrain/multitile'
name = 'merged_downsampled_4x_autofluo.tif'
dx = 5.26
dz = 5
n_downsample = 4

# Merge multi-tile acquisition using maximum strategy
merger = tileMerger(os.path.join(path, 'registration_results_autofluo.csv'), frame_size=2048, type='apr', n_planes=2008)
merger.set_downsample(n_downsample)
merger.initialize_merged_array()
merger.merge_max(mode='constant')

# # Display merged data
fig, ax = plt.subplots(1, 3)
ax[0].imshow(merger.merged_data[250], cmap='gray')
ax[0].set_title('YX')
ax[1].imshow(merger.merged_data[:, 250, :], cmap='gray')
ax[1].set_title('ZX')
ax[2].imshow(merger.merged_data[:, :, 250], cmap='gray')
ax[2].set_title('ZY')

# Equalize histogram
merger.equalize_hist(method='opencv')

# Crop data
# with napari.gui_qt():
#     viewer = napari.Viewer()
#     viewer.add_image(merger.merged_data, contrast_limits=[0, 3000])
merger.crop(background=167, ylim=[0, 733])

imsave(os.path.join(path, name), merger.merged_data)

# Old stuff not necessary since brainreg can do the down-sampling automatically.
# # Interpolate on 25 µm grid
# x = np.linspace(0, merger.merged_data.shape[2]*dx*n_downsample, merger.merged_data.shape[2])
# y = np.linspace(0, merger.merged_data.shape[1]*dx*n_downsample, merger.merged_data.shape[1])
# z = np.linspace(0, merger.merged_data.shape[0]*dz*n_downsample, merger.merged_data.shape[0])
#
# xi = np.arange(0, x.max(), 25)
# yi = np.arange(0, y.max(), 25)
# zi = np.arange(0, z.max(), 25)
# Zi, Yi, Xi = np.meshgrid(zi, yi, xi, indexing='ij')
#
# interpolating_function = RegularGridInterpolator((z, y, x), merger.merged_data)
# data_interp = interpolating_function((Zi, Yi, Xi))
# imsave(os.path.join(path, 'merged_downsampled_25um.tif'), data_interp.astype('uint16'))