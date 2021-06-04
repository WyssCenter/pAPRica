"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import numpy as np
from pipapr.parser import tileParser
from pipapr.loader import tileLoader
import napari
import sparse
from pipapr.viewer import apr_to_napari_Image

# Parameters
path = '/mnt/Data/MC_PV_LOC000_20210503_172011/APR'

# We load a tile
tiles = tileParser(path, frame_size=2048, overlap=512, ftype='apr')
tile = tileLoader(tiles[2])
tile.load_tile()

# We create a sparse array that supports inserting data (COO does not)
labels_sparse = sparse.DOK(shape=(1167, 2048, 2048), dtype='uint8')

# We call napari with the APRSlicer and the sparse array for storing the manual annotations
viewer = napari.Viewer()
viewer.add_layer(apr_to_napari_Image(tile.apr, tile.parts))
viewer.add_labels(labels_sparse)
napari.run()

# We extract labels and pixel coordinate from the sparse array
labels_sparse = labels_sparse.to_coo()
pixel_list = labels_sparse.coords.T
labels = labels_sparse.data

