"""
Script for obtaining the stitching speed against computational ratio.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

# We restrict the number of cores to the optimal setup.
import os
os.environ['OMP_NUM_THREADS'] = '8'

import pipapr
from time import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# Parameters
output_folder_apr = r'/home/hbm/Desktop/data/synthetic/APR'

#
folders = glob(os.path.join(output_folder_apr, '*/'))

elapsed_time = []
cr = []

for folder in folders:
    # Parse data
    tiles = pipapr.parser.tileParser(folder, frame_size=512, ftype='apr')

    # Stitch tiles
    stitcher = pipapr.stitcher.tileStitcher(tiles, overlap_h=25, overlap_v=25)
    stitcher.set_overlap_margin(2)

    t = time()
    stitcher.compute_registration_fast()
    elapsed_time.append(time()-t)

    tmp= []
    for tile in tiles:
        tile.load_tile()
        tmp.append(tile.apr.computational_ratio())
    cr.append(np.mean(tmp))

    print('CR {} - Elapsed time {} s.'.format(cr[-1], elapsed_time[-1]))

plt.plot(cr, elapsed_time, 'k+', label='APR')
plt.plot([np.min(cr), np.max(cr)], [52, 52], 'r:', label='TeraStitcher multicore')
plt.xlabel('Computational ratio')
plt.ylabel('Stitching time [s.]')
plt.title('Stitching benchmark with 4x4 (512x512x2048) tile')
plt.legend()
plt.xscale('log')
plt.yscale('log')