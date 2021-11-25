"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
from time import time
import napari
import os

# Convert data to APR
# path = '/media/hbm/SSD2/rainbow_computer/Lamy_028143_LOC000_20211001_151504/VW0'
# for c, ip in zip([0, 1], [120, 200]):
#     tiles = pipapr.parser.tileParser(path, frame_size=2048, ncol=6, nrow=8, ftype='tiff2D', channel=c)
#     converter = pipapr.converter.tileConverter(tiles)
#     # converter.set_compression(bg=1000)
#     converter.batch_convert_to_apr(Ip_th=ip, rel_error=0.2, gradient_smoothing=2, path='/media/hbm/SSD2/rainbow_computer/Lamy_028143_LOC000_20211001_151504/APR_ch' + str(c))

# Stitch data
path = '/media/hbm/SSD2/rainbow_computer/Lamy_028143_LOC000_20211001_151504/APR_ch1'
tiles = pipapr.parser.tileParser(path, frame_size=2048, ftype='apr')

stitcher = pipapr.stitcher.tileStitcher(tiles, overlap_v=29.005, overlap_h=32.626)
stitcher.set_overlap_margin(5)
# stitcher.set_regularization(reg_x=20, reg_y=20, reg_z=20)
# stitcher.compute_registration_fast()
stitcher.compute_registration_from_max_projs()
stitcher.save_database(os.path.join(path, 'registration_results.csv'))

# viewer = pipapr.viewer.tileViewer(tiles, stitcher.database)
# viewer.check_stitching(downsample=1, contrast_limits=[0, 5000], blending='additive')
stitcher.reconstruct_slice(z=2500, downsample=1, debug=True, plot=True)

# stitcher2 = pipapr.stitcher.tileStitcher(tiles, overlap_v=29.005, overlap_h=32.626)
# stitcher2.compute_expected_registration()
# stitcher2.reconstruct_slice(z=1000, downsample=1, debug=True, plot=True)

# viewer = pipapr.viewer.tileViewer(tiles=tiles, database=stitcher.database)
# viewer.display_tiles([(1,0), (1,1), (0,0), (0,1)], contrast_limits=[0, 3000])

#
# path = '/media/hbm/SSD2/rainbow_computer/Lamy_028143_LOC000_20211001_151504/APR_ch1'
# tiles = pipapr.parser.tileParser(path, frame_size=2048, ftype='apr')

# import pyapr
# for tile in tiles:
#     tile.load_tile()
#     tree_parts = pyapr.ShortParticles()
#     pyapr.numerics.fill_tree_mean(tile.apr, tile.parts, tree_parts)
#     pyapr.io.write(tile.path, tile.apr, tile.parts, tree_parts=tree_parts)

# stitcher = pipapr.stitcher.tileStitcher(tiles, overlap_v=30, overlap_h=33)
# stitcher.set_overlap_margin(2)
# stitcher.compute_registration_fast()
# stitcher.save_database(os.path.join(path, 'registration_results.csv'))

# Display some tiles
# viewer = pipapr.viewer.tileViewer(tiles, stitcher.database)
# viewer.display_tiles(coords=[(6, 1), (6, 2), (7, 1), (7, 2)])

# Merge for checking that everything went fine
# database = os.path.join(path, 'registration_results.csv')
# dqtq = pipapr.stitcher.reconstruct_middle_frame(tiles, database, downsample=1)
# viewer = pipapr.viewer.tileViewer(tiles, database)
# viewer.check_stitching(downsample=32, contrast_limits=[0, 1000])
# merger = pipapr.stitcher.tileMerger(tiles, database, n_planes=1861)
# merger.set_downsample(8)
# merger.merge_max()
#
# viewer = napari.Viewer()
# viewer.add_image(merger.merged_data, scale=[3, 1, 1])
# napari.run()