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
import cv2 as cv

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

class tileCells():

    def __init__(self,
                 tiles: tileParser,
                 database: (str, pd.DataFrame)):

        # If database is a path then load database, if it's a DataFrame keep it as it is.
        if isinstance(database, str):
            self.database = pd.read_csv(database)
        elif isinstance(database, pd.DataFrame):
            self.database = database
        else:
            raise TypeError('Error: database of wrong type.')

        self.tiles = tiles
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

        self.cells = None

    def extract_and_merge_cells(self, lowe_ratio=0.7, distance_max=5):
        
        for t in self.tiles:
            tile = tileLoader(t)
            tile.load_tile()
            tile.load_segmentation()
            apr, parts = tile.data
            apr, cc = tile.data_segmentation

            # Initialized merged cells for the first tile
            if self.cells is None:
                self.cells = pyapr.numerics.transform.find_label_centers(apr, cc, parts)
                self.cells += self._get_tile_position(tile.row, tile.col)
            else:
                self._merge_cells(tile, apr, cc, parts, lowe_ratio=lowe_ratio, distance_max=distance_max)


    def _merge_cells(self, tile, apr, cc, parts, lowe_ratio, distance_max):

        r1 = np.max(self.cells, axis=0)
        r2 = self._get_tile_position(tile.row, tile.col)

        v_size = np.array([apr.org_dims(2), apr.org_dims(1), apr.org_dims(0)])

        # Define the overlapping area
        overlap_i = r2
        overlap_f = np.min((r1 + v_size, r2 + v_size), axis=0)

        # Retrieve cell centers
        cells2 = pyapr.numerics.transform.find_label_centers(apr, cc, parts)
        cells2 += r2

        # Filter cells to keep only those on the overlapping area
        for i in range(3):
            if i == 0:
                ind = np.where(self.cells[:, i] < overlap_i[i])[0]
            else:
                ind = np.concatenate((ind, np.where(self.cells[:, i] < overlap_i[i])[0]))
            ind = np.concatenate((ind, np.where(self.cells[:, i] > overlap_f[i])[0]))
        ind = np.unique(ind)

        cells1_out = self.cells[ind, :]
        cells1_overlap = np.delete(self.cells, ind, axis=0)

        for i in range(3):
            if i == 0:
                ind = np.where(cells2[:, i] < overlap_i[i])[0]
            else:
                ind = np.concatenate((ind, np.where(cells2[:, i] < overlap_i[i])[0]))
            ind = np.concatenate((ind, np.where(cells2[:, i] > overlap_f[i])[0]))
        ind = np.unique(ind)

        cells2_out = cells2[ind, :]
        cells2_overlap = np.delete(cells2, ind, axis=0)

        cells_filtered_overlap = self._filter_cells_flann(cells1_overlap,
                                                          cells2_overlap,
                                                          lowe_ratio=lowe_ratio,
                                                          distance_max=distance_max)

        self.cells = np.vstack((cells1_out, cells2_out, cells_filtered_overlap))

    def _get_tile_position(self, row, col):
        """
        Parse tile position in the database.
        """
        df = self.database
        tile_df = df[(df['row'] == row) & (df['col'] == col)]
        px = tile_df['ABS_H'].values[0]
        py = tile_df['ABS_V'].values[0]
        pz = tile_df['ABS_D'].values[0]

        return np.array([pz, py, px])

    def _filter_cells_flann(self, c1, c2, lowe_ratio=0.7, distance_max=5, verbose=False):

        if lowe_ratio < 0 or lowe_ratio > 1:
            raise ValueError('Lowe ratio is {}, expected between 0 and 1.'.format(lowe_ratio))

        # Match cells descriptors by using Flann method
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=100)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(np.float32(c1), np.float32(c2), k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < lowe_ratio*n.distance and m.distance < distance_max:
                good.append(m)

        # Remove cells that are present in both volumes
        ind_c1 = [m.queryIdx for m in good]
        ind_c2 = [m.trainIdx for m in good]

        # For now I just remove thee cells in c but merging strategies can be better
        c2 = np.delete(c2, ind_c2, axis=0)

        # Display info
        if verbose:
            print('{:0.2f}% of cells were removed.'.format(len(ind_c2)/(c1.shape[0]+c2.shape[0]-len(ind_c2))*100))

        return np.vstack((c1, c2))
