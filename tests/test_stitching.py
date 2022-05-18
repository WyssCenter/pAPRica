"""
Test script for stitching pipeline.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from time import time
import numpy as np
import pipapr
import pandas as pd

# Parameters
path = r'./data/apr/'

# Parse data
tiles = pipapr.parser.tileParser(path, frame_size=512, ftype='apr')

# Stitch data
t = time()
stitcher1 = pipapr.stitcher.tileStitcher(tiles, overlap_h=25, overlap_v=25)
stitcher1.compute_registration(on_disk=False)
print('Elapsed time new registration on RAM: {} s.'.format((time()-t)))
t = time()
stitcher2 = pipapr.stitcher.tileStitcher(tiles, overlap_h=25, overlap_v=25)
stitcher2.compute_registration(on_disk=True)
print('Elapsed time new registration on disk: {} s.'.format((time()-t)))

# Verify that both registration are the same and that it worked
pd.testing.assert_frame_equal(stitcher1.database, stitcher2.database)
assert(stitcher1.effective_overlap_h < 28)
assert(stitcher1.effective_overlap_h > 22)
assert(stitcher2.effective_overlap_h < 28)
assert(stitcher2.effective_overlap_h > 22)

c_graph_from = [2, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 11, 12, 13, 14]
assert(stitcher1.cgraph_from == c_graph_from)
assert(stitcher2.cgraph_from == c_graph_from)

c_graph_to = [3, 6, 7, 5, 8, 6, 9, 7, 11, 9, 12, 13, 15, 13, 14, 15]
assert(stitcher1.cgraph_to == c_graph_to)
assert(stitcher2.cgraph_to == c_graph_to)

assert(len(stitcher1.database) == tiles.n_tiles)
assert(stitcher1.nrow == tiles.nrow)
assert(stitcher1.ncol == tiles.ncol)
assert(stitcher1.frame_size == tiles.frame_size)
assert(stitcher1.n_edges == tiles.n_edges)
assert(stitcher1.n_vertex == tiles.n_tiles)