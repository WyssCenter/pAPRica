import os
from glob import glob
import matplotlib.pyplot as plt
import pipapr
import numpy as np
import re

# def tile_number_to_row_col(nrow, ncol):
#     rows = []
#     cols = []
#     tile_pattern = np.zeros((nrow, ncol), dtype='uint16')
#     row = 0
#     wait = True
#     direction = 1
#     for i in range(nrow*ncol):
#         if i == 0:
#             rows.append(row)
#             cols.append(0)
#         else:
#             rows.append(row)
#             if wait:
#                 cols.append(cols[-1])
#             else:
#                 cols.append(cols[-1] + direction)
#
#         tile_pattern[rows[-1], cols[-1]] = i
#
#         if (cols[-1] == ncol-1) and not wait:
#             direction = -1
#             wait = True
#             row += 1
#         elif (cols[-1] == ncol-1) and wait:
#             wait = False
#         if (cols[-1] == 0) and not wait:
#             wait = True
#             direction = 1
#             row += 1
#         elif (cols[-1] == 0) and wait:
#             wait = False
#     return tile_pattern, rows, cols
#
# files = sorted(glob('/media/hbm/HDD_data/CS_lamy/full_apr/*.apr'), key= lambda x: int(re.findall('(\d+).apr', x)[-1]))
# tile_pattern, rows, cols = tile_number_to_row_col(nrow=17, ncol=13)
#
# for file, row, col in zip(files, rows, cols):
#     os.rename(file, os.path.join(os.path.dirname(file), '{}_{}'.format(row, col)))

tiles = pipapr.parser.tileParser('/media/hbm/HDD_data/CS_lamy/test')
# converter = pipapr.converter.tileConverter(tiles)
# converter.batch_convert_to_apr(Ip_th=170, rel_error=0.2, lazy_loading=True, path='/media/hbm/HDD_data/CS_lamy/full_apr')

stitcher = pipapr.stitcher.tileStitcher(tiles, overlap_h=40, overlap_v=40)
stitcher.set_overlap_margin(2)
stitcher.set_regularization(reg_x=25, reg_y=25, reg_z=25)
stitcher.compute_registration_fast()
# stitcher.save_database('/media/hbm/HDD_data/CS_lamy/full_apr/registration_full_depth_reg_25.csv')
stitcher.reconstruct_slice(loc=100, color=True, n_proj=0, downsample=4)

viewer = pipapr.viewer.tileViewer(tiles, stitcher.database)
viewer.check_stitching(color=True)
viewer.display_tiles(coords=[(5, 2), (5, 3)], color=True, contrast_limits=[0, 2000])

merger = pipapr.stitcher.tileMerger(tiles, stitcher.database)
merger.lazy = False
merger.set_downsample(8)
merger.merge_max()
merger.merged_data[merger.merged_data<150] = 0
merger.crop(ylim=[0, 2189])

atlaser = pipapr.atlaser.tileAtlaser.from_merger(merger=merger, original_pixel_size=[5, 1, 1])
atlaser.register_to_atlas(output_dir='/media/hbm/HDD_data/CS_lamy/atlas', orientation='sal', debug=True)


stitcher_th = pipapr.stitcher.tileStitcher(tiles, overlap_h=40, overlap_v=40)
stitcher_th.compute_expected_registration()

viewer = pipapr.viewer.tileViewer(tiles, stitcher.database)
viewer.display_tiles(coords=[(5, x) for x in range(7)], color=True, contrast_limits=[0, 2000])
#
# stitcher2 = pipapr.stitcher.tileStitcher(tiles, overlap_h=40, overlap_v=40)
# stitcher2.set_overlap_margin(2)
# # stitcher2.compute_registration_fast(on_disk=True)
# stitcher2.compute_registration_from_max_projs()
# # viewer = pipapr.viewer.tileViewer(tiles, stitcher2.database)
# # viewer.check_stitching(blending='additive')
#
# pipapr.viewer.compare_stitching(100, stitcher2, stitcher, downsample=4)

