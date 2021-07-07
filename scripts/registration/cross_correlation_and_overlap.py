"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from time import time

import pipapr
import os
import numpy as np
import pyapr
import napari
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation


# # Parameters
path = '/home/apr-benchmark/PycharmProjects/apr_pipelines/tests/data/apr/'

overlaps = range(100, 510, 20)
over_h = []
over_v = []
relia_h = []
relia_v = []
for overlap in overlaps:
    # Parse data
    tiles = pipapr.parser.tileParser(path, frame_size=512, overlap=overlap, ftype='apr')

    # Stitch and segment
    stitcher = pipapr.stitcher.tileStitcher(tiles)
    stitcher.compute_registration_fast()

    over_h.append(stitcher.effective_overlap_h)
    over_v.append(stitcher.effective_overlap_v)
    relia_h.append(stitcher.graph_relia_H.sum()/16)
    relia_v.append(stitcher.graph_relia_V.sum()/16)

import matplotlib.pyplot as plt

plt.plot(overlaps, relia_h, 'o', label='H')
plt.plot(overlaps, relia_v, 'o', label='V')
plt.legend()
plt.ylabel('Phase cross correlation error normalized')
plt.xlabel('Overlap [pixel]')
plt.figure()
plt.plot(overlaps, over_h, 'o', label='H')
plt.plot(overlaps, over_v, 'o', label='V')
plt.xlabel('Overlap [pixel]')
plt.ylabel('Computed overlap [%]')
#
# tiles = pipapr.parser.tileParser(path, overlap=128, frame_size=512)
# tile = tiles[3]
# tile.load_tile()
# proj1 = pipapr.stitcher._get_max_proj_apr(tile.apr, tile.parts, patch=pyapr.ReconPatch(), plot=True)
# tile = tiles[7]
# tile.load_tile()
# proj2 = pipapr.stitcher._get_max_proj_apr(tile.apr, tile.parts, patch=pyapr.ReconPatch(), plot=True)
#
# reference_image = proj1[2]
# moving_image = proj2[2]
#
# # images must be the same shape
# if reference_image.shape != moving_image.shape:
#     raise ValueError("images must be same shape")
#
# # compute fourier transform
# src_freq = np.fft.fftn(reference_image)
# # src_freq = np.fft.fftn(np.pad(reference_image, pad_width=512))
# target_freq = np.fft.fftn(moving_image)
# # target_freq = np.fft.fftn(np.pad(moving_image, pad_width=512))
#
# # Whole-pixel shift - Compute cross-correlation by an IFFT
# shape = src_freq.shape
# image_product = src_freq * target_freq.conj()
# image_product /= np.abs(image_product)
# cross_correlation = np.fft.ifftn(image_product)
#
# # Locate maximum
# maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
#                           cross_correlation.shape)
# midpoints = np.array([np.fix(axis_size * 0.5) for axis_size in shape])
#
# shifts = np.stack(maxima).astype(np.float64)
# shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
#
# src_amp = np.sum(np.real(src_freq * src_freq.conj()))
# src_amp /= src_freq.size
# target_amp = np.sum(np.real(target_freq * target_freq.conj()))
# target_amp /= target_freq.size
# CCmax = np.abs(cross_correlation[maxima])
#
# # If its only one row or column the shift along that dimension has no
# # effect. We set to zero.
# for dim in range(src_freq.ndim):
#     if shape[dim] == 1:
#         shifts[dim] = 0
#
# error = np.sqrt(1 - CCmax**2 / (reference_image.size * moving_image.size))
#
#
# viewer = napari.Viewer()
# viewer.add_image(reference_image, colormap='green', blending='additive', opacity=0.7)
# viewer.add_image(moving_image, translate=shifts, colormap='red', blending='additive', opacity=0.7)
# napari.run()