"""
Script to process the 10x10 tiles acquired on the COLM.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from time import time
from pipapr.parser import tileParser
from pipapr.loader import tileLoader
from pipapr.stitcher import tileGraph
from pipapr.viewer import tileViewer
from alive_progress import alive_bar
import os
import numpy as np
import matplotlib.pyplot as plt

path = r'/home/jules/Desktop/mouse_colm/multitile'
t = time()
t_ini = time()
tiles = tileParser(path, frame_size=2048, overlap=int(2048*0.25), type='apr')
print('Elapsed time parse data: {:.2f} ms.'.format((time() - t) * 1000))
t = time()
tgraph = tileGraph(tiles)
print('Elapsed time init tgraph: {:.2f} ms.'.format((time() - t) * 1000))
t = time()
with alive_bar(len(tiles), force_tty=True) as bar:
    for tile in tiles:
        loaded_tile = tileLoader(tile)
        loaded_tile.compute_registration(tgraph)
        # loaded_tile.activate_mask(threshold=95)
        # loaded_tile.compute_segmentation(path_classifier=
        # r'/media/sf_shared_folder_virtualbox/PV_interneurons/classifiers/random_forest_n100.joblib')
        bar()
print('Elapsed time load, segment, and compute pairwise reg: {:.2f} s.'.format(time() - t))

t = time()
tgraph.build_sparse_graphs()
print('Elapsed time build sparse graph: {:.2f} ms.'.format((time() - t) * 1000))
t = time()
tgraph.optimize_sparse_graphs()
print('Elapsed time optimize graph: {:.2f} ms.'.format((time() - t) * 1000))
tgraph.plot_min_trees(annotate=True)
t = time()
reg_rel_map, reg_abs_map = tgraph.produce_registration_map()
print('Elapsed time reg map: {:.2f} ms.'.format((time() - t) * 1000))
t = time()
tgraph.build_database(tiles)
print('Elapsed time build database: {:.2f} ms.'.format((time() - t) * 1000))
t = time()
tgraph.save_database(os.path.join(path, 'registration_results.csv'))
print('Elapsed time save database: {:.2f} ms.'.format((time() - t) * 1000))

print('\n\nTOTAL elapsed time: {:.2f} s.'.format(time() - t_ini))

viewer = tileViewer(tiles, tgraph, segmentation=False)
coords = []
for i in range(4, 6):
    for j in range(4, 6):
        coords.append([i, j])
coords = np.array(coords)
viewer.display_tiles(coords, level_delta=0, contrast_limits=[0, 1000])


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

# # Equalize histogram
# merger.equalize_hist(method='opencv')
#
# # Crop data
# with napari.gui_qt():
#     viewer = napari.Viewer()
#     viewer.add_image(merger.merged_data, contrast_limits=[0, 10000], scale=[3/0.59, 1, 1])


# merger.crop(background=167, ylim=[0, 733])

# imsave(os.path.join(path, 'merged_8x_clahe.tif'), merger.merged_data)