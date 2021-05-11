"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pandas as pd
from pipapr.parser import tileParser
from pipapr.loader import tileLoader
import os
import numpy as np
import pyapr
from joblib import load
from time import time
from pathlib import Path

class tileSegmenter():

    def __init__(self,
                 tile: tileLoader,
                 path_classifier, func_to_compute_features, func_to_get_cc):
        """

        Parameters
        ----------
        tile: (tileLoader) tile object for loading the tile (or containing the preloaded tile).
        path_classifier: (str) path to pre-trained classifier
        func_to_compute_features: (func) function to compute the features on ParticleData. Must be the same set of
                                        as the one used to train the classifier.
        func_to_get_cc: (func) function to post process the segmentation map into a connected component (each cell has
                                        a unique id)
        """

        self.path = tile.path
        self.data = tile.data
        self.is_tile_loaded = tile.data is not None
        # Load classifier
        self.clf = load(path_classifier)
        # Store function to compute features
        self.func_to_compute_features = func_to_compute_features
        # Store post processing steps
        self.func_to_get_cc = func_to_get_cc

    def _predict_on_APR_block(self, x, n_parts=1e7, verbose=False):
        """
        Predict particle class with the trained classifier clf on the precomputed features f using a
        blocked strategy to avoid memory segfault.
        """
        # Predict on numpy array by block to avoid memory issues
        if verbose:
            t = time()

        y_pred = np.empty((x.shape[0]))
        n_block = int(np.ceil(x.shape[0] / n_parts))
        if int(n_parts) != n_parts:
            raise ValueError('Error: n_parts must be an int.')
        n_parts = int(n_parts)

        self.clf[1].set_params(n_jobs=-1)
        for i in range(n_block):
            y_pred[i * n_parts:min((i + 1) * n_parts, x.shape[0])] = self.clf.predict(
                x[i * n_parts:min((i + 1) * n_parts, x.shape[0])])

        if verbose:
            print('Blocked prediction took {:0.3f} s.\n'.format(time() - t))

        # Transform numpy array to ParticleData
        parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

        return parts_pred

    def compute_segmentation(self, verbose=False):
        """
        Compute the segmentation and stores the result as an independent APR.

        Parameters
        ----------
        verbose: (bool) control the verbosity of the function to print some info

        Returns
        -------
        None
        """

        apr = self.data[0]
        parts = self.data[1]

        if verbose:
            t = time()
            print('Computing features on AP')
        f = self.func_to_compute_features(apr, parts)
        if verbose:
            print('Features computation took {:0.2f} s.'.format(time()-t))

        parts_pred = self._predict_on_APR_block(f, verbose=verbose)

        if verbose:
            # Display inference info
            print('\n****** INFERENCE RESULTS ******')
            print(
                '{} cell particles ({:0.2f}%)'.format(np.sum(parts_pred == 0),
                                                      np.sum(parts_pred == 0) / len(parts_pred) * 100))
            print('{} background particles ({:0.2f}%)'.format(np.sum(parts_pred == 1),
                                                              np.sum(parts_pred == 1) / len(parts_pred) * 100))
            print('{} membrane particles ({:0.2f}%)'.format(np.sum(parts_pred == 2),
                                                            np.sum(parts_pred == 2) / len(parts_pred) * 100))
            print('*******************************')

        cc = self.func_to_get_cc(apr, parts_pred)

        folder, filename = os.path.split(self.path)
        folder_seg = os.path.join(folder, 'segmentation')
        Path(folder_seg).mkdir(parents=True, exist_ok=True)
        pyapr.io.write(os.path.join(folder_seg, filename[:-4] + '_segmentation.apr'), apr, cc)

class cellTile():

    def __init__(self,
                 tiles: tileParser,
                 database: (str, pd.DataFrame)):

        # If database is a path then load dabatase, if it's a DataFrame keep it as it is.
        if isinstance(database, str):
            self.database = pd.read_csv(database)
        elif isinstance(database, pd.DataFrame):
            self.database = database
        else:
            raise TypeError('Error: database of wrong type.')

        self.path = tiles.path
        self.type = tiles.type
        self.tiles_list = tiles.tiles_list
        self.n_tiles = tiles.n_tiles
        self.ncol = tiles.ncol
        self.nrow = tiles.nrow
        self.neighbors = tiles.neighbors
        self.n_edges = tiles.n_edges
        self.path_list = tiles.path_list
        self.overlap = tiles.overlap
        self.frame_size = tiles.frame_size

    def extract_and_merge_cells(self):
        
        for t in self.tiles:
            tile = tileLoader(t)

            for v, coords in zip(tile.data_neighbors, tile.neighbors):
                if tile.row == coords[0] and tile.col < coords[1]:
                    # EAST
                    reg, rel = tile._compute_east_registration(v)
    
                elif tile.col == coords[1] and tile.row < coords[0]:
                    # SOUTH
                    reg, rel = tile._compute_south_registration(v)
    
                else:
                    raise TypeError('Error: couldn''t determine merging to perform.')