"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
import napari
import os

# Convert data to APR
path = '/media/hbm/SSD1/mesoSPIM/Lamy_028245'
# # PV
# tiles = pipapr.parser.randomParser(os.path.join(path, 'ch0'), frame_size=2048, ftype='raw')
# converter = pipapr.converter.tileConverter(tiles)
# converter.batch_convert_to_apr(Ip_th=140, rel_error=0.2, gradient_smoothing=2)
# # Autofluo
# tiles = pipapr.parser.randomParser(os.path.join(path, 'ch1'), frame_size=2048, ftype='raw')
# converter = pipapr.converter.tileConverter(tiles)
# converter.batch_convert_to_apr(Ip_th=300, rel_error=0.2, gradient_smoothing=2)

# Stitch data
path_signal = '/media/hbm/SSD1/mesoSPIM/Lamy_028245/APR/ch0'
tiles_signal = pipapr.parser.tileParser(path_signal, frame_size=2048, ftype='apr')
stitcher = pipapr.stitcher.tileStitcher(tiles_signal, overlap_h=58, overlap_v=28)
# stitcher.set_overlap_margin(1)
# stitcher.compute_registration_fast()
# stitcher.save_database(os.path.join(path, 'registration_results.csv'))
database = os.path.join(path, 'registration_results.csv')

path_autofluo = '/media/hbm/SSD1/mesoSPIM/Lamy_028245/APR/ch1'
tiles_autofluo = pipapr.parser.tileParser(path_autofluo, frame_size=2048, ftype='apr')
# stitcher_autofluo = pipapr.stitcher.channelStitcher(stitcher, tiles_autofluo, tiles_signal)
# stitcher_autofluo = pipapr.stitcher.tileStitcher(tiles_autofluo, overlap_h=58, overlap_v=28)
# stitcher_autofluo.compute_registration_fast()
# stitcher_autofluo.compute_rigid_registration()

# v = pipapr.viewer.tileViewer(tiles_signal, database)
# v.display_all_tiles()

# Merge and atlas autofluo data
merger = pipapr.stitcher.tileMerger(tiles_autofluo, database, n_planes=1837)
merger.set_downsample(4)
merger.merge_max()
# import matplotlib.pyplot as plt
# import numpy as np
# plt.imshow(np.max(merger.merged_data, axis=2), cmap='gray')
merger.merged_data[merger.merged_data<220] = 0
# napari.view_image(merger.merged_data)
merger.crop(ylim=[121, merger.merged_data.shape[1]])
atlaser = pipapr.atlaser.tileAtlaser.from_merger(merger, original_pixel_size=[5*2, 5.26*2, 5.26*2])

params = {'affine-n-steps': 4,
         'affine-use-n-steps': 3}

atlaser.register_to_atlas(output_dir='/media/hbm/SSD1/mesoSPIM/Lamy_028245/', orientation='spr', debug=True,
                          params=params)


# viewer_autofluo = pipapr.viewer.tileViewer(tiles_autofluo, database)
# l = viewer_autofluo.get_layers_all_tiles(colormap='green', contrast_limits=[0, 1000])
# viewer_signal = pipapr.viewer.tileViewer(tiles_signal, database)
# l.extend(viewer_signal.get_layers_all_tiles(colormap='red', contrast_limits=[0, 20000]))
# pipapr.viewer.display_layers(l)