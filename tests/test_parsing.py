"""
Test script for parsing correctly multi-tile data-sets.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""


import paprica
import numpy as np
import os


def test_main():
    # Parameters
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'apr')

    # Parse data
    tiles = paprica.parser.tileParser(path, frame_size=512, ftype='apr')
    tiles.check_files_integrity()
    tiles.compute_average_CR()

    # Verify that parsing is correct
    assert(tiles.n_tiles == 13)
    assert(tiles.ncol == 4)
    assert(tiles.nrow == 4)
    assert(tiles.n_edges == 21)
    assert(tiles.frame_size == 512)
    assert(tiles.path == path)
    assert(tiles.type == 'apr')

    neighbors = np.array([[list([[1, 0]]), list([[0, 2], [1, 1]]), list([[0, 3], [1, 2]]),
            list([[1, 3]])],
           [list([[1, 1], [2, 0]]), list([[1, 2], [2, 1]]), list([[1, 3]]),
            list([[2, 3]])],
           [list([[2, 1], [3, 0]]), list([[3, 1]]), list([[2, 3], [3, 2]]),
            list([[3, 3]])],
           [list([[3, 1]]), list([[3, 2]]), list([[3, 3]]), list([])]],
          dtype=object)
    assert((tiles.neighbors == neighbors).all())

    neighbors_tot = np.array([[list([[1, 0]]), list([[0, 2], [1, 1]]), list([[0, 3], [1, 2]]),
            list([[1, 3], [0, 2]])],
           [list([[1, 1], [2, 0]]), list([[1, 2], [2, 1], [1, 0]]),
            list([[1, 3], [1, 1], [0, 2]]), list([[2, 3], [1, 2], [0, 3]])],
           [list([[2, 1], [3, 0], [1, 0]]), list([[3, 1], [2, 0], [1, 1]]),
            list([[2, 3], [3, 2], [2, 1], [1, 2]]), list([[3, 3], [1, 3]])],
           [list([[3, 1], [2, 0]]), list([[3, 2], [3, 0], [2, 1]]),
            list([[3, 3], [3, 1]]), list([[3, 2], [2, 3]])]], dtype=object)
    assert((tiles.neighbors_tot == neighbors_tot).all())

    path_list = [os.path.join(path, '0_2.apr'),
                 os.path.join(path, '0_3.apr'),
                 os.path.join(path, '1_0.apr'),
                 os.path.join(path, '1_1.apr'),
                 os.path.join(path, '1_2.apr'),
                 os.path.join(path, '1_3.apr'),
                 os.path.join(path, '2_0.apr'),
                 os.path.join(path, '2_1.apr'),
                 os.path.join(path, '2_3.apr'),
                 os.path.join(path, '3_0.apr'),
                 os.path.join(path, '3_1.apr'),
                 os.path.join(path, '3_2.apr'),
                 os.path.join(path, '3_3.apr')]
    assert(tiles.path_list == path_list)

    tiles_pattern = np.array([[0., 0., 1., 1.],
                               [1., 1., 1., 1.],
                               [1., 1., 0., 1.],
                               [1., 1., 1., 1.]])
    assert((tiles.tiles_pattern == tiles_pattern).all())

    tile_pattern_path = np.array([[None, None, os.path.join(path, '0_2.apr'), os.path.join(path, '0_3.apr')],
                                   [os.path.join(path, '1_0.apr'), os.path.join(path, '1_1.apr'), os.path.join(path, '1_2.apr'),
                                    os.path.join(path, '1_3.apr')],
                                   [os.path.join(path, '2_0.apr'), os.path.join(path, '2_1.apr'), None,
                                    os.path.join(path, '2_3.apr')],
                                   [os.path.join(path, '3_0.apr'), os.path.join(path, '3_1.apr'), os.path.join(path, '3_2.apr'),
                                    os.path.join(path, '3_3.apr')]], dtype=object)
    assert((tiles.tile_pattern_path == tile_pattern_path).all())