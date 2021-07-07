"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
from time import time
import napari
import pandas as pd


path = '/run/user/1000/gvfs/smb-share:server=wyssnasc.campusbiotech.ch,share=computingdata/Alice/9007_CBT_PNP_LIGHTSHEET/Jules/223_Petersen/VW0/'

tiles = pipapr.parser.tileParser(path, overlap=28, frame_size=2048, ncol=10, nrow=9, ftype='tiff2D')
converter = pipapr.converter.tileConverter(tiles)
converter.set_compression(bg=1000)
converter.batch_convert_to_apr(Ip_th=1000, rel_error=0.2, gradient_smoothing=0, path='/home/apr-benchmark/Desktop/data/petersen_2')

# path = '/home/apr-benchmark/Desktop/data/petersen'
#
# tiles = pipapr.parser.tileParser(path, overlap=30, frame_size=2048)
# stitcher = pipapr.stitcher.tileStitcher(tiles)
# t = time()
# stitcher.compute_registration_fast()
# print('Elapsed time stitching: {} s.'.format(time()-t))
# stitcher.save_database('/home/apr-benchmark/Desktop/data/petersen/registration_results.csv')
#
# merger = pipapr.stitcher.tileMerger(tiles, stitcher.database, n_planes=2000)
# merger.set_downsample(downsample=16)
# merger.merge_max()
#
# napari.view_image(merger.merged_data)
#
# viewer = pipapr.viewer.tileViewer(tiles, database=pd.read_csv('/home/apr-benchmark/Desktop/data/petersen/registration_results.csv'))
# viewer.display_tiles(coords=[(5,5), (6,5), (5,6), (6,6)], contrast_limits=[5000, 2**16])