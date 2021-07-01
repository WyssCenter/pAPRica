"""
Script to try out napari

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from napari.layers import Image
import napari
import pyapr
import pipapr
import pandas as pd

def get_pyramid(apr, parts, n):
    p = []
    for i in range(n):
        p.append(pyapr.data_containers.APRSlicer(apr, parts, mode='constant', level_delta=-i))
    return p

# Parameters
path = '../../tests/data/apr'

# Load APR file
tiles = pipapr.parser.tileParser(path, frame_size=2048, overlap=512, ftype='apr')
database = pd.read_csv('../../tests/data/registration_results.csv')

# Display pyramidal file with Napari
# tile = tiles[3]
# tile.load_tile()
# layers = [pipapr.viewer.apr_to_napari_Image(apr=tile.apr, parts=tile.parts, level_delta=0)]
# pipapr.viewer.display_layers_pyramidal(layers, level_delta=0)

viewer = pipapr.viewer.tileViewer(tiles, database)
viewer.display_all_tiles(pyramidal=True)