"""
Module containing classes and functions relative to Segmentation.


By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import os
from time import time

import cv2 as cv
import napari
import numpy as np
import pandas as pd
import pyapr
import sparse
from joblib import load
from tqdm import tqdm

import paprica


def _predict_on_APR_block(x, clf, n_parts=1e7, output='class', verbose=False):
    """
    Predict particle class with the trained classifier clf on the precomputed features f using a
    blocked strategy to avoid memory segfault.

    Parameters
    ----------
    x: ndarray
        features (n_particle, n_features) for particle prediction
    n_parts: int
        number of particles in the batch to predict
    output: string
        output type, can be 'class' where each particle get assigned a class or 'proba' where each
        particle get assigned a probability of belonging to each class.
    verbose: bool
        control function verbosity

    Returns
    -------
    parts_pred: array_like
        Class prediction for each particle.
    """

    # Predict on numpy array by block to avoid memory issues
    if verbose:
        t = time()

    n_block = int(np.ceil(x.shape[0] / n_parts))
    if int(n_parts) != n_parts:
        raise ValueError('Error: n_parts must be an int.')
    n_parts = int(n_parts)
    clf[1].set_params(n_jobs=-1)

    if output == 'class':
        y_pred = np.empty((x.shape[0]))
        for i in tqdm(range(n_block), desc='Predicting particle type'):
            y_pred[i * n_parts:min((i + 1) * n_parts, x.shape[0])] = clf.predict(
                x[i * n_parts:min((i + 1) * n_parts, x.shape[0])])
        # Transform numpy array to ParticleData
        parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

    elif output == 'proba':
        y_pred = np.empty((x.shape[0], len(clf.classes_)))
        for i in tqdm(range(n_block), desc='Predicting particle type'):
            y_pred[i * n_parts:min((i + 1) * n_parts, x.shape[0]), :] = clf.predict_proba(
                x[i * n_parts:min((i + 1) * n_parts, x.shape[0])])
        # Transform numpy array to ParticleData
        parts_pred = []
        for i in range(len(clf.classes_)):
            parts_pred.append(pyapr.ShortParticles(
                                                   (y_pred[:, i]*(2**16-1))
                                                    .astype('uint16')))
    else:
        raise ValueError('Unknown output \'{}\' for APR block prediction.'.format(output))

    if verbose:
        print('Blocked prediction took {:0.3f} s.\n'.format(time() - t))


    return parts_pred


def map_feature(apr, parts_cc, features):
    """
    Map feature values to segmented particle data.

    Parameters
    ----------
    apr: pyapr.APR
        apr object to map features to
    parts_cc: pyapr.ParticleData
        connected component particle array corresponding to apr
    features: array_like
        array containing the values to map

    Returns
    -------
    Array of mapped values (each particle in the connected component now has the value present in features)
    """

    objects_volume = pyapr.measure.find_label_volume(apr, parts_cc)
    hash_idx = np.arange(0, len(objects_volume))
    # Object with volume 0 are not in CC so we need to get rid of them
    hash_idx = hash_idx[objects_volume > 0]
    # We also need to get rid of the background
    hash_idx = hash_idx[1:]

    if len(hash_idx) != len(features):
        raise ValueError('Error: features length ({}) should be the same as the number of connected components ({}).'
                         .format(len(features), len(hash_idx)))

    # Create hash dict
    hash_dict = {x: y for x, y in zip(hash_idx, features)}
    # Replace 0 by 0
    hash_dict[0] = 0

    mp = np.arange(0, parts_cc.max() + 1)
    mp[list(hash_dict.keys())] = list(hash_dict.values())
    return mp[np.array(parts_cc, copy=False)]


def compute_gradients(apr, parts):
    """
    Compute gradient for each spatial direction directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    Returns
    -------
    (dx, dy, dz): (arrays) gradient for each direction
    """

    par = apr.get_parameters()

    dz = pyapr.filter.gradient(apr, parts, dim=2, delta=par.dz)
    dx = pyapr.filter.gradient(apr, parts, dim=1, delta=par.dx)
    dy = pyapr.filter.gradient(apr, parts, dim=0, delta=par.dy)
    return dz, dx, dy


def compute_laplacian(apr, parts, grad=None):
    """
    Compute Laplacian for each spatial direction directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    grad: (dz, dy, dx) gradient for each direction if precomputed (faster for Laplacian computation)

    Returns
    -------
    Laplacian of APR.
    """

    par = apr.get_parameters()
    if grad is None:
        dz, dx, dy = compute_gradients(apr, parts)
    else:
        dz, dx, dy = grad
    dz2 = pyapr.filter.gradient(apr, dz, dim=2, delta=par.dz)
    dx2 = pyapr.filter.gradient(apr, dx, dim=1, delta=par.dx)
    dy2 = pyapr.filter.gradient(apr, dy, dim=0, delta=par.dy)
    return dz2 + dx2 + dy2


def compute_gradmag(apr, parts):
    """
    Compute gradient magnitude directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR

    Returns
    -------
    Gradient magnitude of APR.
    """

    par = apr.get_parameters()
    gradmag = pyapr.filter.gradient_magnitude(apr, parts, deltas=(par.dz, par.dx, par.dy))
    return gradmag


def gaussian_blur(apr, parts, sigma=1.5, size=11):
    """
    Compute Gaussian blur directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    sigma: (float) Gaussian blur standard deviation (kernel radius)
    size: (int) kernel size (increase with caution, complexity is not linear)

    Returns
    -------
    Blurred APR.
    """

    stencil = pyapr.filter.get_gaussian_stencil(size, sigma, ndims=3, normalize=True)
    return pyapr.filter.convolve(apr, parts, stencil)


def particle_levels(apr):
    """
    Returns apr level: for each particle the lvl is defined as the size of the particle in pixel.

    Parameters
    ----------
    apr: (APR) APR object

    Returns
    -------
    Particle level.
    """

    lvls = pyapr.ShortParticles(apr.total_number_particles())
    lvls.fill_with_levels(apr)
    lvls = np.array(lvls)

    return 2 ** (lvls.max() - lvls)


class tileSegmenter():
    """
    Class used to segment tiles. It is instantiated with a tileLoader object, a previously trained classifier,
    a function to compute features (the same features used to train the classifier and a function to get the
    post processed connected component for the classifier output.

    """

    def __init__(self, clf, func_to_compute_features, func_to_get_cc, verbose):
        """

        Parameters
        ----------
        clf: sklearn.classifier
            pre-trained classifier
        func_to_compute_features: func
            function to compute the features on ParticleData. Must be the same set of
            as the one used to train the classifier.
        func_to_get_cc: func
            function to post process the segmentation map into a connected component (each cell has
                                        a unique id)
        """

        # Store classifier
        self.clf = clf
        # Store function to compute features
        self.func_to_compute_features = func_to_compute_features
        # Store post processing steps
        self.func_to_get_cc = func_to_get_cc
        # Verbose
        self.verbose = verbose

    @classmethod
    def from_trainer(cls,
                     trainer,
                     verbose=True):
        """
        Instantiate tileSegmenter object with a tileTrainer object.

        Parameters
        ----------
        trainer: tileTrainer
            trainer object previously trained for segmentation
        verbose: bool
            control function output

        Returns
        -------
        tileSegmenter object
        """

        return cls(clf=trainer.clf,
                   func_to_compute_features=trainer.func_to_compute_features,
                   func_to_get_cc=trainer.func_to_get_cc,
                   verbose=verbose)

    @classmethod
    def from_classifier(cls,
                        classifier,
                        func_to_compute_features,
                        func_to_get_cc=None,
                        verbose=True):
        """
        Instantiate tileSegmenter object with a classifier, function to compute the features and to get the
        connected components.

        Parameters
        ----------
        classifier
        func_to_compute_features: func
            function to compute features used by the classifier to perform the segmentation.
        func_to_get_cc: func
            function to compute the connected component from the classifier prediction.
        verbose: bool
            control function output.

        Returns
        -------
        tileSegmenter object
        """

        if isinstance(classifier, str):
            clf = load(classifier)
        else:
            clf = classifier

        return cls(clf=clf,
                   func_to_compute_features=func_to_compute_features,
                   func_to_get_cc=func_to_get_cc,
                   verbose=verbose)

    def compute_segmentation(self, tile: paprica.loader.tileLoader,
                             save_cc=True, save_mask=False, lazy_loading=True):
        """
        Compute the segmentation and stores the result as an independent APR.

        Parameters
        ----------
        verbose: bool
            control the verbosity of the function to print some info

        Returns
        -------
        None
        """

        if tile.apr is None:
            tile.load_tile()

        # Compute features on APR
        if self.verbose:
            t = time()
            print('Computing features on APR')
        f = self.func_to_compute_features(tile.apr, tile.parts)
        # self.filtered_APR = f
        if self.verbose:
            print('Features computation took {:0.2f} s.'.format(time()-t))

        # Predict particle class
        parts_pred = _predict_on_APR_block(f, self.clf, verbose=self.verbose)
        if self.verbose:
            # Display inference info
            print('\n****** INFERENCE RESULTS ******')
            for l in self.clf.classes_:
                print('Class {}: {} particles ({:0.2f}%)'.format(l, np.sum(parts_pred == l),
                                                      np.sum(parts_pred == l) / len(parts_pred) * 100))
            print('*******************************')

        # Compute connected component from classification
        if self.func_to_get_cc is not None:
            cc = self.func_to_get_cc(tile.apr, parts_pred)
            tile.parts_cc = cc

        # Save results
        if save_mask:
            # Write particles
            pyapr.io.write_particles(tile.path, parts_pred, parts_name='segmentation mask', tree=False, append=True)
            if lazy_loading:
                # Compute tree parts)
                tree_parts = pyapr.tree.fill_tree_max(tile.apr, parts_pred)
                # Save tree parts
                pyapr.io.write_particles(tile.path, tree_parts, parts_name='segmentation mask', tree=True, append=True)
        if save_cc:
            # Write particles
            pyapr.io.write_particles(tile.path, cc, parts_name='segmentation cc', tree=False, append=True)
            if lazy_loading:
                # Compute tree parts
                tree_parts = pyapr.tree.fill_tree_max(tile.apr, cc)
                # Save tree parts
                pyapr.io.write_particles(tile.path, tree_parts, parts_name='segmentation cc', tree=True, append=True)

        tile.parts_mask = parts_pred

    # def _save_segmentation(self, path, name, parts):
    #     """
    #     Save segmentation particles by appending the original APR file.
    #
    #     Parameters
    #     ----------
    #     parts: pyapr.ParticleData
    #         particles to save. Note that the APR tree should be the same otherwise the data
    #         will be inconsistent and not readable.
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     pyapr.io.write_particles(path, parts, parts_name=name, append=True)
    #     pyapr.io.write_particles(path, tree_parts, parts_name=name, append=True, tree=True)
    #     # aprfile = pyapr.io.APRFile()
    #     # aprfile.set_read_write_tree(True)
    #     # aprfile.open(path, 'READWRITE')
    #     # aprfile.write_particles(name, parts, t=0)
    #     # aprfile.close()


class multitileSegmenter():
    """
    Class used to segment multitiles acquisition.
    """

    def __init__(self,
                 tiles: paprica.parser.tileParser,
                 database: (str, pd.DataFrame),
                 clf,
                 func_to_compute_features,
                 func_to_get_cc,
                 verbose=True):
        """

        Parameters
        ----------
        tiles: tileLoader
            tile object for loading the tile (or containing the preloaded tile).
        database: pd.DataFrame, string
            dataframe (or path to the csv file) containing the registration parameters to correctly place each tile.
        clf: sklearn.classifier
            pre-trained classifier
        func_to_compute_features: func
            function to compute the features on ParticleData. Must be the same set of
            as the one used to train the classifier.
        func_to_get_cc: func
            function to post process the segmentation map into a connected component (each cell has
                                        a unique id)
        """

        # Store classifier
        if isinstance(clf, str):
            self.clf = load(clf)
        else:
            self.clf = clf
        # Store function to compute features
        self.func_to_compute_features = func_to_compute_features
        # Store post processing steps
        self.func_to_get_cc = func_to_get_cc

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
        self.frame_size = tiles.frame_size
        self.verbose = verbose

        self.cells = None
        self.atlas = None


    @classmethod
    def from_trainer(cls,
                     tiles: paprica.parser.tileParser,
                     database: (str, pd.DataFrame),
                     trainer,
                     verbose=True):
        """
        Instantiate tileSegmenter object with a tileTrainer object.

        Parameters
        ----------
        trainer: tileTrainer
            trainer object previously trained for segmentation
        verbose: bool
            control function output

        Returns
        -------
        tileSegmenter object
        """

        return cls(tiles=tiles,
                   database= database,
                   clf=trainer.clf,
                   func_to_compute_features=trainer.func_to_compute_features,
                   func_to_get_cc=trainer.func_to_get_cc,
                   verbose=verbose)

    @classmethod
    def from_classifier(cls,
                        tiles: paprica.parser.tileParser,
                        database: (str, pd.DataFrame),
                        classifier,
                        func_to_compute_features,
                        func_to_get_cc=None,
                        verbose=True):
        """
        Instantiate tileSegmenter object with a classifier, function to compute the features and to get the
        connected components.

        Parameters
        ----------
        classifier
        func_to_compute_features: func
            function to compute features used by the classifier to perform the segmentation.
        func_to_get_cc: func
            function to compute the connected component from the classifier prediction.
        verbose: bool
            control function output.

        Returns
        -------
        tileSegmenter object
        """

        if isinstance(classifier, str):
            clf = load(classifier)
        else:
            clf = classifier

        return cls(tiles=tiles,
                   database= database,
                   clf=clf,
                   func_to_compute_features=func_to_compute_features,
                   func_to_get_cc=func_to_get_cc,
                   verbose=verbose)


    def compute_multitile_segmentation(self, save_cc=True, save_mask=False, lowe_ratio=0.7, distance_max=5, lazy_loading=True):
        """
        Compute the segmentation and stores the result as an independent APR.

        Parameters
        ----------
        verbose: bool
            control the verbosity of the function to print some info
        save_cc: bool
            option to save the connected component particle to file
        save_mask: bool
            option to save the prediction mask to file
        lowe_ratio: float in ]0, 1[
            ratio between the second nearest neighbor and the first nearest neighbor to be considered a good match
        distance_max: float
            maximum distance in pixel for two objects to be matched
        lazy_loading: bool
            option to save the tree particles to allow for lazy loading later on

        Returns
        -------
        None
        """

        for tile in tqdm(self.tiles, desc='Extracting and merging cells..'):

            # Perform tile segmentation
            tile = self._segment_tile(tile, save_cc=save_cc, save_mask=save_mask, lazy_loading=lazy_loading)

            # Remove objects on the edge
            pyapr.morphology.remove_edge_objects(tile.apr, tile.parts_cc, z_edges=False)

            # Initialized merged cells for the first tile
            if self.cells is None:
                self.cells = pyapr.measure.find_label_centers(tile.apr, tile.parts_cc, tile.parts)
                self.cells += self._get_tile_position(tile.row, tile.col)
            # Then merge the rest on the first tile
            else:
                self._merge_cells(tile, lowe_ratio=lowe_ratio, distance_max=distance_max)

            try:
                self.save_cells(output_path=os.path.join(self.tiles.path, 'cells_backup.csv'))
            except:
                print('Could not back up cells.')

    def extract_and_merge_cells(self, lowe_ratio=0.7, distance_max=5):
        """
        Function to extract cell positions in each tile and merging across all tiles.
        Identical cells on overlapping area are automatically detected using Flann method.

        Parameters
        ----------
        lowe_ratio: float
            ratio of the second nearest neighbor distance / nearest neighbor distance above lowe_ratio, the cell is
            supposed to be unique. Below lowe_ratio, it might have a second detection on the neighboring tile.
        distance_max: float
            maximum distance in pixel for two cells to be considered the same.

        Returns
        -------
        None
        """
        
        for tile in tqdm(self.tiles, desc='Extracting and merging cells..'):
            tile.load_tile()
            tile.load_segmentation()
            
            # Remove objects on the edge
            pyapr.morphology.remove_edge_objects(tile.apr, tile.parts_cc)

            # Initialized merged cells for the first tile
            if self.cells is None:
                self.cells = pyapr.measure.find_label_centers(tile.apr, tile.parts_cc, tile.parts)
                self.cells += self._get_tile_position(tile.row, tile.col)
            # Then merge the rest on the first tile
            else:
                self._merge_cells(tile, lowe_ratio=lowe_ratio, distance_max=distance_max)

            try:
                self.save_cells(output_path=os.path.join(self.tiles.path, 'cells_backup.csv'))
            except:
                print('Could not back up cells.')

    def save_cells(self, output_path):
        """
        Save cells as a CSV file.

        Parameters
        ----------
        output_path: string
            path for saving the CSV file.

        Returns
        -------
        None
        """

        pd.DataFrame(self.cells).to_csv(output_path, header=['z', 'y', 'x'])

    def _segment_tile(self, tile: paprica.loader.tileLoader,
                      save_cc=True, save_mask=False, lazy_loading=True):
        """
        Compute the segmentation and stores the result as an independent APR.

        Parameters
        ----------
        verbose: bool
            control the verbosity of the function to print some info

        Returns
        -------
        None
        """

        if tile.apr is None:
            tile.load_tile()

        # Compute features on APR
        if self.verbose:
            t = time()
            print('Computing features on APR')
        f = self.func_to_compute_features(tile.apr, tile.parts)
        self.filtered_APR = f
        if self.verbose:
            print('Features computation took {:0.2f} s.'.format(time()-t))

        # Predict particle class
        parts_pred = _predict_on_APR_block(f, self.clf, verbose=self.verbose)
        if self.verbose:
            # Display inference info
            print('\n****** INFERENCE RESULTS ******')
            for l in self.clf.classes_:
                print('Class {}: {} particles ({:0.2f}%)'.format(l, np.sum(parts_pred == l),
                                                      np.sum(parts_pred == l) / len(parts_pred) * 100))
            print('*******************************')

        # Compute connected component from classification
        if self.func_to_get_cc is not None:
            cc = self.func_to_get_cc(tile.apr, parts_pred)
            tile.parts_cc = cc

        # Save results
        if save_mask:
            # Write particles
            pyapr.io.write_particles(tile.path, parts_pred, parts_name='segmentation mask', tree=False, append=True)
            if lazy_loading:
                # Compute tree parts)
                tree_parts = pyapr.tree.fill_tree_max(tile.apr, parts_pred)
                # Save tree parts
                pyapr.io.write_particles(tile.path, tree_parts, parts_name='segmentation mask', tree=True, append=True)
        if save_cc:
            # Write particles
            pyapr.io.write_particles(tile.path, cc, parts_name='segmentation cc', tree=False, append=True)
            if lazy_loading:
                # Compute tree parts
                tree_parts = pyapr.tree.fill_tree_max(tile.apr, cc)
                # Save tree parts
                pyapr.io.write_particles(tile.path, tree_parts, parts_name='segmentation cc', tree=True, append=True)

        tile.parts_mask = parts_pred

        return tile

    def _merge_cells(self, tile, lowe_ratio, distance_max):
        """
        Function to merge cells on a tile to the final cells list and remove duplicate.

        Parameters
        ----------
        tile: tileLoader
            tile from which to merge cells
        lowe_ratio: float
            ratio of the second nearest neighbor distance / nearest neighbor distance above lowe_ratio, the cell is
            supposed to be unique. Below lowe_ratio, it might have a second detection on the neighboring tile.
        distance_max: float
            maximum distance in pixel for two cells to be considered the same.

        Returns
        -------
        None
        """

        if self.cells.shape[0]>0:
            r1 = np.max(self.cells, axis=0)
        else:
            r1 = np.array([0, 0, 0])
        r2 = self._get_tile_position(tile.row, tile.col)

        v_size = np.array(tile.apr.shape())

        # Define the overlapping area
        overlap_i = r2
        overlap_f = np.min((r1 + v_size, r2 + v_size), axis=0)

        # Retrieve cell centers
        cells2 = pyapr.measure.find_label_centers(tile.apr, tile.parts_cc, tile.parts)
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

        # Number of cells should be higher than the number of features for the KNN implementation to work.
        if cells2_overlap.shape[0] > 4:
            cells_filtered_overlap = self._filter_cells_flann(cells1_overlap,
                                                              cells2_overlap,
                                                              lowe_ratio=lowe_ratio,
                                                              distance_max=distance_max)

            self.cells = np.vstack((cells1_out, cells2_out, cells_filtered_overlap))
        else:
            self.cells = np.vstack((cells1_out, cells2_out))

    def _get_tile_position(self, row, col):
        """
        Function to get the absolute tile position defined by it's coordinate in the multitile set.

        Parameters
        ----------
        row: int
            row number
        col: int
            column number

        Returns
        -------
        _: ndarray
            tile absolute position
        """

        df = self.database
        tile_df = df[(df['row'] == row) & (df['col'] == col)]
        px = tile_df['ABS_H'].values[0]
        py = tile_df['ABS_V'].values[0]
        pz = tile_df['ABS_D'].values[0]

        return np.array([pz, py, px])

    def _filter_cells_flann(self, c1, c2, lowe_ratio=0.7, distance_max=5):
        """
        Remove cells duplicate using Flann criteria and distance threshold.

        Parameters
        ----------
        c1: ndarray
            array containing the first set cells coordinates
        c2: ndarray
            array containing the second set cells coordinates
        lowe_ratio: float
            ratio of the second nearest neighbor distance / nearest neighbor distance above lowe_ratio, the cell is
            supposed to be unique. Below lowe_ratio, it might have a second detection on the neighboring tile.
        distance_max: float
            maximum distance in pixel for two cells to be considered the same.
        verbose: bool
            control function verbosity

        Returns
        -------
        _: ndarray
            array containing the merged sets without the duplicates.
        """

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

        # For now I just remove the cells in c2 but merging strategies can be better
        c2 = np.delete(c2, ind_c2, axis=0)

        # Display info
        if self.verbose:
            try:
                print('{:0.2f}% of cells were removed.'.format(len(ind_c2)/(c1.shape[0]+c2.shape[0]-len(ind_c2))*100))
            except ZeroDivisionError:
                print('No cell removed.')

        return np.vstack((c1, c2))

    # def _save_segmentation(self, path, name, parts, tree_parts):
    #     """
    #     Save segmentation particles by appending the original APR file.
    #
    #     Parameters
    #     ----------
    #     parts: pyapr.ParticleData
    #         particles to save. Note that the APR tree should be the same otherwise the data
    #         will be inconsistent and not readable.
    #     tree_parts: pyapr.ParticleData
    #         tree particles necessary for lazy loading.
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     pyapr.io.write_particles(path, parts, parts_name=name, append=True)
    #     pyapr.io.write_particles(path, tree_parts, parts_name=name, append=True, tree=True)
    #     # aprfile = pyapr.io.APRFile()
    #     # aprfile.set_read_write_tree(True)
    #     # aprfile.open(path, 'READWRITE')
    #     # aprfile.write_particles(name, parts, t=0, tree_parts=tree_parts)
    #     # aprfile.close()


class tileTrainer():
    """
    Class used to train a classifier that works directly on APR data. It uses Napari to manually add labels.

    """

    def __init__(self,
                 tile: paprica.loader.tileLoader,
                 func_to_compute_features,
                 func_to_get_cc=None):

        tile.load_tile()
        self.tile = tile
        self.apr = tile.apr
        self.parts = tile.parts
        self.apr_it = self.apr.iterator()
        self.shape = tile.apr.shape()
        self.func_to_compute_features = func_to_compute_features
        self.func_to_get_cc = func_to_get_cc

        self.labels_manual = None
        self.pixel_list = None
        self.labels = None
        self.use_sparse_labels = None
        self.parts_train_idx = None
        self.clf = None
        self.parts_mask = None
        self.parts_cc = None
        self.f = None

    def manually_annotate(self, use_sparse_labels=True, **kwargs):
        """
        Manually annotate dataset using Napari.

        Parameters
        ----------
        use_sparse_labels: bool
            use sparse array to store the labels (memory efficient but slower graphics)

        Returns
        -------
        None
        """
        self.sparse = use_sparse_labels

        if self.sparse:
            # We create a sparse array that supports inserting data (COO does not)
            self.labels_manual = sparse.DOK(shape=self.shape, dtype='uint8')
        else:
            self.labels_manual = np.empty(self.shape, dtype='uint8')

        # We call napari with the APRSlicer and the sparse array for storing the manual annotations
        viewer = napari.Viewer()
        image_layer = napari.layers.Image(data=pyapr.reconstruction.APRSlicer(self.apr, self.parts), **kwargs)
        viewer.add_layer(image_layer)
        viewer.add_labels(self.labels_manual)
        viewer.show(block=True)

        # We extract labels and pixel coordinate from the sparse array
        if self.sparse:
            self.labels_manual = self.labels_manual.to_coo()
        else:
            self.labels_manual = sparse.COO.from_numpy(self.labels_manual)

        self.pixel_list = self.labels_manual.coords.T
        self.labels = self.labels_manual.data

    def add_annotations(self, use_sparse_labels=True, **kwargs):
        """
        Add annotations on previously annotated dataset.

        Parameters
        ----------
        use_sparse_labels: bool
            use sparse array to store the labels (memory efficient but slower graphics)

        Returns
        -------
        None
        """

        self.sparse = use_sparse_labels

        if self.sparse:
            # We create a sparse array that supports inserting data (COO does not)
            self.labels_manual = sparse.DOK(self.labels_manual)
        else:
            self.labels_manual = self.labels_manual.todense()

        # We call napari with the APRSlicer and the sparse array for storing the manual annotations
        viewer = napari.Viewer()
        image_layer = napari.layers.Image(data=pyapr.reconstruction.APRSlicer(self.apr, self.parts), **kwargs)
        viewer.add_layer(image_layer)
        viewer.add_labels(self.labels_manual)
        viewer.show(block=True)

        # We extract labels and pixel coordinate from the sparse array
        if self.sparse:
            self.labels_manual = self.labels_manual.to_coo()
        else:
            self.labels_manual = sparse.COO.from_numpy(self.labels_manual)

        self.pixel_list = self.labels_manual.coords.T
        self.labels = self.labels_manual.data

    def save_labels(self, path=None):
        """
        Save labels as numpy array with columns corresponding to [z, y, x, label].

        Parameters
        ----------
        path: string
            path to save labels. By default it saves them in the data root folder.

        Returns
        -------
        None
        """

        if path is None:
            path = os.path.join(self.tile.folder_root, 'manual_labels.npy')

        to_be_saved = np.hstack((self.pixel_list, self.labels[:, np.newaxis]))
        np.save(path, to_be_saved)

    def load_labels(self, path=None):
        """
        Load previously saved labels as numpy array with columns corresponding to [z, y, x, label].

        Parameters
        ----------
        path: string
            path to load the saved labels. By default it loads them in the data root folder.

        Returns
        -------
        None
        """
        if path is None:
            path = os.path.join(self.tile.folder_root, 'manual_labels.npy')

        data = np.load(path)
        self.pixel_list = data[:, :-1]
        self.labels = data[:, -1]
        self.labels_manual = sparse.COO(coords=self.pixel_list.T, data=self.labels, shape=self.shape)

    def train_classifier(self, verbose=True, n_estimators=10, class_weight='balanced',
                         mean_norm=True, std_norm=True):
        """
        Train the classifier for segmentation.

        Parameters
        ----------
        verbose: bool
            option to print out information.
        n_estimators : int
            The number of trees in the random forest.
        class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts,
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.

            Note that for multioutput (including multilabel) weights should be
            defined for each class of every column in its own dict. For example,
            for four-class multilabel classification weights should be
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
            [{1:1}, {2:5}, {3:1}, {4:1}].

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``

            The "balanced_subsample" mode is the same as "balanced" except that
            weights are computed based on the bootstrap sample for every tree
            grown.

            For multi-output, the weights of each column of y will be multiplied.

            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.
        mean_norm : bool
            If True, center the data before scaling.
        std_norm : bool
            If True, scale the data to unit variance (or equivalently,
            unit standard deviation).

        Returns
        -------
        None
        """
        if self.pixel_list is None:
            raise ValueError('Error: annotate dataset or load annotations before training classifier.')

        from sklearn import preprocessing
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestClassifier

        # We sample pixel_list on APR grid
        self._sample_pixel_list_on_APR()

        # We remove ambiguous case where a particle was labeled differently
        self._remove_ambiguities(verbose=verbose)

        # We compute features and train the classifier
        if self.f is None:
            self.f = self.func_to_compute_features(self.apr, self.parts)

        # Fetch data that was manually labelled
        x = self.f[self.parts_train_idx]
        y = self.parts_labels

        # Train random forest
        clf = make_pipeline(preprocessing.StandardScaler(with_mean=mean_norm, with_std=std_norm),
                            RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight))
        t = time()
        clf.fit(x, y.ravel())
        print('Training took {} s.\n'.format(time() - t))

        x_pred = clf.predict(x)

        # Display training info
        if verbose:
            print('\n****** TRAINING RESULTS ******')
            print('Total accuracy: {:0.2f}%'.format(np.sum(x_pred == y) / y.size * 100))
            for l in self.unique_labels:
                print('Class {} accuracy: {:0.2f}% ({} cell particles)'.format(l,
                    np.sum((x_pred == y) * (y == l)) / np.sum(y == l) * 100, np.sum(y == l)))
            print('******************************\n')

        self.clf = clf

    def segment_training_tile(self, bg_label=None, display_result=True, verbose=True):
        """
        Apply classifier to the whole tile and display segmentation results using Napari.

        Parameters
        ----------
        display_result: bool
            option to display segmentation results using Napari
        verbose: bool
            option to print out information.

        Returns
        -------
        None
        """

        # Apply on whole dataset
        if self.parts_mask is None:
            parts_pred = _predict_on_APR_block(self.f, self.clf, verbose=verbose)
            self.parts_mask = parts_pred

        if (self.func_to_get_cc is not None) and self.parts_cc is None:
            self.parts_cc = self.func_to_get_cc(self.apr, self.parts_mask.copy())

        # Display inference info
        if verbose:
            print('\n****** INFERENCE RESULTS ******')
            for l in self.unique_labels:
                print('Class {}: {} cell particles ({:0.2f}%)'.format(l, np.sum(self.parts_mask == l),
                                                            np.sum(self.parts_mask == l) / len(self.parts_mask) * 100))
            print('******************************\n')

        # Display segmentation using Napari
        if display_result:
            if self.parts_cc is not None:
                paprica.viewer.display_segmentation(self.apr, self.parts, self.parts_cc)
            elif bg_label is not None:
                parts_pred = np.array(self.parts_mask.copy())
                parts_pred[parts_pred == bg_label] = 0
                parts_pred = pyapr.ShortParticles(parts_pred)
                paprica.viewer.display_segmentation(self.apr, self.parts, parts_pred)

    def display_training_annotations(self, **kwargs):
        """
        Display manual annotations and their sampling on APR grid (if available).

        Returns
        -------
        None
        """
        image_nap = napari.layers.Image(data=pyapr.reconstruction.APRSlicer(self.apr, self.parts),
                                        opacity=0.7, **kwargs)
        viewer = napari.Viewer()
        viewer.add_layer(image_nap)
        viewer.add_labels(self.labels_manual, name='Manual labels', opacity=0.5)
        if self.parts_labels is not None:
            mask = np.zeros_like(self.parts, dtype='uint16')
            mask[self.parts_train_idx] = self.parts_labels
            label_map = napari.layers.Labels(data=pyapr.reconstruction.APRSlicer(self.apr,
                                                                                 pyapr.ShortParticles(mask),
                                                                                 tree_mode='max'),
                                             name='APR labels',
                                             opacity=0.5)
            viewer.add_layer(label_map)
        napari.run()

    def apply_on_tile(self, tile, bg_label=None, func_to_get_cc=None, display_result=True, verbose=True):
        """
        Apply classifier to the whole tile and display segmentation results using Napari.

        Parameters
        ----------
        display_result: bool
            option to display segmentation results using Napari
        verbose: bool
            option to print out information.

        Returns
        -------
        None
        """

        # Apply on whole dataset
        if tile.apr is None:
            tile.load_tile()
        f = self.func_to_compute_features(tile.apr, tile.parts)
        parts_pred = _predict_on_APR_block(f, self.clf, verbose=verbose)
        tile.parts_mask = parts_pred.copy()

        # Display inference info
        if verbose:
            print('\n****** INFERENCE RESULTS ******')
            for l in self.unique_labels:
                print('Class {}: {} cell particles ({:0.2f}%)'.format(l, np.sum(parts_pred == l),
                                                            np.sum(parts_pred == l) / len(parts_pred) * 100))
            print('******************************\n')

        # Display segmentation using Napari
        if display_result:
            if func_to_get_cc is not None:
                tile.parts_cc = func_to_get_cc(tile.apr, parts_pred)
                paprica.viewer.display_segmentation(tile.apr, tile.parts, tile.parts_cc)
            elif bg_label is not None:
                parts_pred = np.array(parts_pred)
                parts_pred[parts_pred==bg_label] = 0
                parts_pred = pyapr.ShortParticles(parts_pred)
                paprica.viewer.display_segmentation(tile.apr, tile.parts, parts_pred)

    def save_classifier(self, path=None):
        """
        Save the trained classifier.

        Parameters
        ----------
        path: string
            path for saving the classifier. By default, it is saved in the data root folder.

        Returns
        -------
        None
        """
        from joblib import dump

        if path is None:
            path = os.path.join(self.tile.folder_root, 'random_forest_n100.joblib')

        dump(self.clf, path)

    def load_classifier(self, path=None):
        """
        Load a trained classifier.

        Parameters
        ----------
        path: string
            path for loading the classifier. By default, it is loaded from root folder.

        Returns
        -------
        None
        """
        from joblib import load

        if path is None:
            path = os.path.join(self.tile.folder_root, 'random_forest_n100.joblib')

        self.clf = load(path)

    def display_features(self):
        """
        Display the computed features.

        """
        if self.f is None:
            raise TypeError('Error: filters can''t be displayed because they were not computed')

        viewer = napari.Viewer()
        for i in range(self.f.shape[1]):
            viewer.add_layer(paprica.viewer.apr_to_napari_Image(self.apr, pyapr.FloatParticles(self.f[:, i])))
        napari.run()

    def _remove_ambiguities(self, verbose):
        """
        Remove particles that have been labelled with different labels.

        Parameters
        ----------
        verbose: bool
            option to print out information.

        Returns
        -------

        """
        if self.parts_train_idx is None:
            raise ValueError('Error: train classifier before removing ambiguities.')

        idx_unique = np.unique(self.parts_train_idx)

        parts_train = []
        parts_labels = []

        cnt = 0
        for idx in idx_unique:
            local_labels = self.labels[self.parts_train_idx==idx]
            is_same, l = self._are_labels_the_same(local_labels)
            if is_same:
                # If labels are the same we assign it
                parts_labels.append(l)
                parts_train.append(idx)
            else:
                cnt += 1

        self.parts_train_idx = np.array(parts_train)
        self.parts_labels = np.array(parts_labels)
        self.unique_labels = np.unique(self.parts_labels)

        if verbose:
            print('\n********* ANNOTATIONS ***********')
            print('{} ambiguous particles were removed.'.format(cnt))
            print('{} particles were labeled.'.format(self.parts_labels.shape[0]))
            for l in self.unique_labels:
                print('{} particles ({:0.2f}%) were labeled as {}.'.format(np.sum(self.parts_labels==l),
                                                                       100*np.sum(self.parts_labels==l)/self.parts_labels.shape[0],
                                                                        l))
            print('***********************************\n')

    def _sample_pixel_list_on_APR(self):
        """
        Convert manual annotations coordinates from pixel to APR.

        Returns
        -------
        None
        """
        self.parts_train_idx = np.empty(self.pixel_list.shape[0], dtype='uint64')

        for i in tqdm(range(self.pixel_list.shape[0]), desc='Sampling labels on APR.'):
            idx = self._find_particle(self.pixel_list[i, :])

            self.parts_train_idx[i] = idx

    def _find_particle(self, coords):
        """
        Find particle index corresponding to pixel location coords.

        Parameters
        ----------
        coords: array_like
            pixel coordinate [z, y, x]

        Returns
        -------
        idx: (int) particle index
        """
        # TODO: @JOEL PUT THIS IN C++ PLEASE
        for level in range(self.apr_it.level_min(), self.apr_it.level_max()+1):
            particle_size = 2 ** (self.apr.level_max() - level)
            z_l, x_l, y_l = coords // particle_size
            for idx in range(self.apr_it.begin(level, z_l, x_l), self.apr_it.end()):
                if self.apr_it.y(idx) == y_l:
                    # if np.sqrt(np.sum((coords-np.array([z_l, x_l, y_l])*particle_size)**2))/particle_size > np.sqrt(3):
                    #     print('ich')
                    return idx

    def _order_labels(self):
        """
        Order pixel_list in z increasing order, then y increasing order and finally x increasing order.

        Returns
        -------
        None
        """
        for d in range(3, 0):
            ind = np.argsort(self.pixel_list[:, d])
            self.pixel_list = self.pixel_list[ind]
            self.labels = self.labels[ind]

    @staticmethod
    def _are_labels_the_same(local_labels):
        """
        Determine if manual labels in particle are the same and return the labels

        Parameters
        ----------
        local_labels: ndarray
            particle labels

        Returns
        -------
        ((bool): True if labels are the same, (int) corresponding label)
        """
        return ((local_labels == local_labels[0]).all(), local_labels[0])