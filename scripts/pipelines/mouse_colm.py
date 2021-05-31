"""
Script to process the 10x10 tiles acquired on the COLM.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from time import time
from pipapr.parser import tileParser
from pipapr.stitcher import tileStitcher, tileMerger
from pipapr.viewer import tileViewer
from pipapr.converter import tileConverter
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imsave

path = r'/home/jules/Desktop/mouse_colm/multitile_auto'
t = time()
t_ini = time()
tiles = tileParser(path, frame_size=2048, overlap=int(2048*0.25), ftype='apr')

t = time()
stitcher = tileStitcher(tiles)
stitcher.compute_registration_fast(on_disk=False)
print('Elapsed time: {} s.'.format(time()-t))


#
# viewer = tileViewer(tiles, tgraph, segmentation=False)
# coords = []
# for i in range(4, 6):
#     for j in range(4, 6):
#         coords.append([i, j])
# coords = np.array(coords)
# viewer.display_tiles(coords, level_delta=0, contrast_limits=[0, 1000])
#

# Merge multi-tile acquisition using maximum strategy
merger = tileMerger(os.path.join(path, 'registration_results.csv'), frame_size=2048, type='apr', n_planes=1835)
merger.set_downsample(8)
merger.initialize_merged_array()
merger.merge_max(mode='constant')

# Display merged data
fig, ax = plt.subplots(1, 3)
ax[0].imshow(merger.merged_data[150], cmap='gray')
ax[0].set_title('YX')
ax[1].imshow(merger.merged_data[:, 1000, :], cmap='gray')
ax[1].set_title('ZX')
ax[2].imshow(merger.merged_data[:, :, 1000], cmap='gray')
ax[2].set_title('ZY')

# Equalize histogram
merger.equalize_hist(method='opencv')

# # Crop data
# with napari.gui_qt():
#     viewer = napari.Viewer()
#     viewer.add_image(merger.merged_data, contrast_limits=[0, 10000], scale=[3/0.59, 1, 1])

merger.crop(background=167, ylim=[0, 733])

imsave(os.path.join(path, 'merged_8x_clahe_auto.tif'), merger.merged_data)