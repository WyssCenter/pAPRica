"""
Module containing classes and functions relative to Stitching.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from scipy.sparse import csr_matrix
import pandas as pd
# import cv2 as cv
from skimage.exposure import equalize_adapthist
from alive_progress import alive_bar
import dill
from pipapr.loader import tileLoader
from pipapr.parser import tileParser
from pipapr.segmenter import tileSegmenter
import matplotlib.pyplot as plt
import pyapr
from skimage.registration import phase_cross_correlation
from scipy.signal import correlate


def _get_max_proj_apr(apr, parts, patch, plot=False):
    """
    Compute maximum projection on 3D APR data.

    Parameters
    ----------
    apr: (pyapr.APR) apr tree
    parts: (pyapr.ParticlData) apr particle
    patch: (pyapr.patch) patch for computing the projection only on the overlapping area.
    plot: (bool) control data plotting

    Returns
    -------
    (list of np.array) maximum intensity projection in each 3 dimension.

    """
    proj = []
    for d in range(3):
        # dim=0: project along Y to produce a ZY plane
        # dim=1: project along X to produce a ZX plane
        # dim=2: project along Z to produce an YX plane
        proj.append(pyapr.numerics.transform.projection.maximum_projection_patch(apr, parts, dim=d, patch=patch))

    if plot:
        fig, ax = plt.subplots(1, 3)
        for i, title in enumerate(['ZY', 'ZX', 'YX']):
            ax[i].imshow(proj[i], cmap='gray')
            ax[i].set_title(title)

    return proj[0], proj[1], proj[2]


def _get_proj_shifts(proj1, proj2, upsample_factor=1):
    """
    This function computes shifts from max-projections on overlapping areas. It uses the phase cross-correlation
    to compute the shifts.

    Parameters
    ----------
    proj1: (list of np.array) max-projections for tile 1
    proj2: (list of np.array) max-projections for tile 2
    upsample_factor: (float) upsampling_factor for estimating the maximum phase cross-correlation position

    Returns
    -------
    shifts in (x, y, z) and error measure (0=reliable, 1=not reliable)

    """
    # Compute phase cross-correlation to extract shifts
    dzy, error_zy, _ = phase_cross_correlation(proj1[0], proj2[0],
                                               return_error=True, upsample_factor=upsample_factor)
    dzx, error_zx, _ = phase_cross_correlation(proj1[1], proj2[1],
                                               return_error=True, upsample_factor=upsample_factor)
    dyx, error_yx, _ = phase_cross_correlation(proj1[2], proj2[2],
                                               return_error=True, upsample_factor=upsample_factor)

    # Keep only the most reliable registration
    # D/z
    if error_zx < error_zy:
        dz = dzx[0]
        rz = error_zx
    else:
        dz = dzy[0]
        rz = error_zy

    # H/x
    if error_zx < error_yx:
        dx = dzx[1]
        rx = error_zx
    else:
        dx = dyx[1]
        rx = error_yx

    # V/y
    if error_yx < error_zy:
        dy = dyx[0]
        ry = error_yx
    else:
        dy = dzy[1]
        ry = error_zy

    # if self.row==0 and self.col==0:
    #     for i, title in enumerate(['ZY', 'ZX', 'YX']):
    #         fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    #         ax[0].imshow(proj1[i], cmap='gray')
    #         ax[0].set_title('dx={}, dy={}, dz={}'.format(dx, dy, dz))
    #         ax[1].imshow(proj2[i], cmap='gray')
    #         ax[1].set_title(title)
    #     print('ok')

    return np.array([dz, dy, dx]), np.array([rz, ry, rx])


def _get_masked_proj_shifts(proj1, proj2, threshold, upsample_factor=1):
    """
    This function computes shifts from max-projections on overlapping areas with mask on brightest area.
    It uses the phase cross-correlation to compute the shifts.

    Parameters
    ----------
    proj1: (list of arrays) max-projections for tile 1
    proj2: (list of arrays) max-projections for tile 2
    upsample_factor: (float) upsampling_factor for estimating the maximum phase cross-correlation position

    Returns
    -------
    shifts in (x, y, z) and error measure (0=reliable, 1=not reliable)

    """
    # Compute mask to discard very bright area that are likely bubbles or artefacts
    mask_ref = []
    mask_move = []
    for i in range(3):
        vmax = np.percentile(proj1[i], threshold)
        mask_ref.append(proj1[i] < vmax)
        vmax = np.percentile(proj2[i], threshold)
        mask_move.append(proj2[i] < vmax)

    # Compute phase cross-correlation to extract shifts
    dzy = phase_cross_correlation(proj1[0], proj2[0],
                                  return_error=True, upsample_factor=upsample_factor,
                                  reference_mask=mask_ref[0], moving_mask=mask_move[0])
    error_zy = _get_registration_error(proj1[0], proj2[0])
    dzx = phase_cross_correlation(proj1[1], proj2[1],
                                  return_error=True, upsample_factor=upsample_factor,
                                  reference_mask=mask_ref[1], moving_mask=mask_move[1])
    error_zx = _get_registration_error(proj1[1], proj2[1])
    dyx = phase_cross_correlation(proj1[2], proj2[2],
                                  return_error=True, upsample_factor=upsample_factor,
                                  reference_mask=mask_ref[2], moving_mask=mask_move[2])
    error_yx = _get_registration_error(proj1[2], proj2[2])

    # Keep only the most reliable registration
    # D/z
    if error_zx < error_zy:
        dz = dzx[0]
        rz = error_zx
    else:
        dz = dzy[0]
        rz = error_zy

    # H/x
    if error_zx < error_yx:
        dx = dzx[1]
        rx = error_zx
    else:
        dx = dyx[1]
        rx = error_yx

    # V/y
    if error_yx < error_zy:
        dy = dyx[0]
        ry = error_yx
    else:
        dy = dzy[1]
        ry = error_zy

    # if self.row==0 and self.col==0:
    #     for i, title in enumerate(['ZY', 'ZX', 'YX']):
    #         fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    #         ax[0].imshow(proj1[i], cmap='gray')
    #         ax[0].set_title('dx={}, dy={}, dz={}'.format(dx, dy, dz))
    #         ax[1].imshow(proj2[i], cmap='gray')
    #         ax[1].set_title(title)
    #     print('ok')

    return np.array([dz, dy, dx]), np.array([rz, ry, rx])


def _get_registration_error(proj1, proj2):
    """
    Return registration error estimate using the peak height.

    Parameters
    ----------
    proj1: (np.array) first projection
    proj2: (np.array) second projection

    Returns
    -------
    (float) registration error estimate

    """
    return np.sqrt(1-correlate(proj1, proj2).max()**2/(np.sum(proj1**2)*np.sum(proj2**2)))


class baseStitcher():
    def __init__(self,
                 tiles: tileParser):

        self.tiles = tiles
        self.ncol = tiles.ncol
        self.nrow = tiles.nrow
        self.n_vertex = tiles.n_tiles
        self.n_edges = tiles.n_edges
        self.overlap = tiles.overlap
        self.frame_size = tiles.frame_size

        self.mask = False
        self.threshold = None

    def activate_mask(self, threshold):
        """
        Activate the masked cross-correlation for the displacement estimation. Pixels above threshold are
        not taken into account.

        Parameters
        ----------
        threshold: (int) threshold for the cross-correlation mask as a percentage of pixel to keep (e.g. 95 will
                    create a mask removing the 5% brightest pixels).

        """
        self.mask = True
        self.threshold = threshold

    def deactivate_mask(self):
        """
        Deactivate the masked cross-correlation and uses a classical cross correlation.

        """
        self.mask = False
        self.threshold = None

    def save_database(self, path):
        """
        Save database at the given path. The database must be built before calling this method.

        Parameters
        ----------
        path: (str) path to save the database.

        """

        if self.database is None:
            raise TypeError('Error: database can''t be saved because it was not created. '
                            'Please call build_database() first.')

        self.database.to_csv(path)



class tileStitcher(baseStitcher):
    """
    Class used to perform the stitching. The stitching is performed in 4 steps:

    1. The pairwise registration parameters of each neighboring tile is computed on the max-projection
    2. A sparse graph (edges = tiles and vertex = registration between neighboring tiles) is constructed to store
       the registration parameters (displacements and reliability)
    3. The sparse graph is optimized to satisfy the constraints (every loop in the graph should sum to 0) using the
       maximum spanning tree on the reliability estimation.
    4. The maximum spanning tree is parsed to extract optimal tile positions solution.

    The beauty of this method is that it scales well with increasing dataset sizes and because the final optimization
    is very fast and does not require to reload the data.

    """
    # TODO: store the max-proj to avoid loading apr data twice.
    def __init__(self,
                 tiles: tileParser):
        """

        Parameters
        ----------
        tiles: (tileParser) tileParser object containing the dataset to stitch.
        """

        super().__init__(tiles)

        self.cgraph_from = []
        self.cgraph_to = []
        self.relia_H = []
        self.relia_V = []
        self.relia_D = []
        self.dH = []
        self.dV = []
        self.dD = []

        # Attributes below are set when the corresponding method are called.
        self.registration_map_rel = None
        self.registration_map_abs = None
        self.ctree_from_H = None
        self.ctree_from_V = None
        self.ctree_from_D = None
        self.ctree_to_H = None
        self.ctree_to_V = None
        self.ctree_to_D = None
        self.min_tree_H = None
        self.min_tree_V = None
        self.min_tree_D = None
        self.graph_relia_H = None
        self.graph_relia_V = None
        self.graph_relia_D = None
        self.database = None

        self.segment = False

    def activate_segmentation(self, path_classifier, func_to_compute_features,
                              func_to_get_cc, verbose=False):
        """
        Activate the segmentation. When a tile is loaded it is segmented before the stitching is done.

        Parameters
        ----------
        tile: (tileLoader) tile object for loading the tile (or containing the preloaded tile).
        path_classifier: (str) path to pre-trained classifier
        func_to_compute_features: (func) function to compute the features on ParticleData. Must be the same set of
                                        as the one used to train the classifier.
        func_to_get_cc: (func) function to post process the segmentation map into a connected component (each cell has
                                        a unique id)
        """

        self.segment = True
        self.segmentation_verbose = verbose
        # Load classifier
        self.path_classifier = path_classifier
        # Store function to compute features
        self.func_to_compute_features = func_to_compute_features
        # Store post processing steps
        self.func_to_get_cc = func_to_get_cc


    def deactivate_segmentation(self):
        """
        Deactivate tile segmentation.

        """

        self.segment = False



    def compute_registration(self):
        """
        Compute the pair-wise registration for a given tile with all its neighbors (EAST and SOUTH to avoid
        the redundancy).

        """
        for t in self.tiles:
            tile = tileLoader(t)
            tile.load_tile()
            tile.load_neighbors()

            if self.segment:
                segmenter = tileSegmenter(tile,
                                          self.path_classifier,
                                          self.func_to_compute_features,
                                          self.func_to_get_cc)
                segmenter.compute_segmentation(verbose=self.segmentation_verbose)

            for v, coords in zip(tile.data_neighbors, tile.neighbors):

                if tile.row == coords[0] and tile.col < coords[1]:
                    # EAST
                    reg, rel = self._compute_east_registration(tile.data, v)

                elif tile.col == coords[1] and tile.row < coords[0]:
                    # SOUTH
                    reg, rel = self._compute_south_registration(tile.data, v)

                else:
                    raise TypeError('Error: couldn''t determine registration to perform.')

                self.cgraph_from.append(np.ravel_multi_index([tile.row, tile.col],
                                                                    dims=(self.nrow, self.ncol)))
                self.cgraph_to.append(np.ravel_multi_index([coords[0], coords[1]],
                                                                  dims=(self.nrow, self.ncol)))
                # H=x, V=y, D=z
                self.dH.append(reg[2])
                self.dV.append(reg[1])
                self.dD.append(reg[0])
                self.relia_H.append(rel[2])
                self.relia_V.append(rel[1])
                self.relia_D.append(rel[0])

        self._build_sparse_graphs()
        self._optimize_sparse_graphs()
        _, _ = self._produce_registration_map()
        self._build_database()

    def plot_graph(self, annotate=False):
        """
        Plot the graph for each direction (H, D, V). This method needs to be called after the graph
        optimization.

        Parameters
        ----------
        annotate: (bool) control if annotation are drawn on the graph

        """

        if self.graph_relia_H is None:
            raise TypeError('Error: graph not build yet, please use build_sparse_graph()'
                            'before trying to plot the graph.')

        fig, ax = plt.subplots(1, 3)
        for i, d in enumerate(['H', 'V', 'D']):
            ind_from = getattr(self, 'cgraph_from')
            row, col = np.unravel_index(ind_from, shape=(self.nrow, self.ncol))
            V1 = np.vstack((row, col)).T

            ind_to = getattr(self, 'cgraph_to')
            row, col = np.unravel_index(ind_to, shape=(self.nrow, self.ncol))
            V2 = np.vstack((row, col)).T

            rel = getattr(self, 'relia_' + d)
            dX = getattr(self, 'd' + d)
            for ii in range(V1.shape[0]):
                ax[i].plot([V1[ii, 1], V2[ii, 1]], [V1[ii, 0], V2[ii, 0]], 'ko', markerfacecolor='r')
                if annotate:
                    p1 = ax[i].transData.transform_point([V1[ii, 1], V1[ii, 0]])
                    p2 = ax[i].transData.transform_point([V2[ii, 1], V2[ii, 0]])
                    dy = p2[1]-p1[1]
                    dx = p2[0]-p1[0]
                    rot = np.degrees(np.arctan2(dy, dx))
                    if rel[ii] < 0.15:
                        color = 'g'
                    elif rel[ii] < 0.30:
                        color = 'orange'
                    else:
                        color = 'r'
                    ax[i].annotate(text='err={:.2f} d{}={:.2f}'.format(rel[ii], d, dX[ii]),
                                   xy=((V1[ii, 1]+V2[ii, 1])/2, (V1[ii, 0]+V2[ii, 0])/2),
                                   ha='center',
                                   va='center',
                                   rotation=rot,
                                   backgroundcolor='w',
                                   color=color)
            ax[i].set_title(d + ' tree')
            ax[i].invert_yaxis()

        return fig, ax

    def plot_min_trees(self, annotate=False):
        """
        Plot the minimum spanning tree for each direction (H, D, V). This method needs to be called after the graph
        optimization.

        Parameters
        ----------
        annotate: (bool) control if annotation are drawn on the graph

        """

        if self.min_tree_H is None:
            raise TypeError('Error: minimum spanning tree not computed yet, please use optimize_sparse_graph()'
                            'before trying to plot the trees.')

        fig, ax = self.plot_graph(annotate=annotate)

        for i, d in enumerate(['H', 'V', 'D']):
            ind_from = getattr(self, 'ctree_from_' + d)
            row, col = np.unravel_index(ind_from, shape=(self.nrow, self.ncol))
            V1 = np.vstack((row, col)).T

            ind_to = getattr(self, 'ctree_to_' + d)
            row, col = np.unravel_index(ind_to, shape=(self.nrow, self.ncol))
            V2 = np.vstack((row, col)).T

            rel = getattr(self, 'relia_' + d)
            dX = getattr(self, 'd' + d)
            for ii in range(V1.shape[0]):
                ax[i].plot([V1[ii, 1], V2[ii, 1]], [V1[ii, 0], V2[ii, 0]], 'ko-', markerfacecolor='r', linewidth=2)
            ax[i].set_title(d + ' tree')

    def plot_registration_map(self):
        """
        Display the registration map using matplotlib.

        """

        if self.registration_map_abs is None:
            raise TypeError('Error: registration map not computed yet, please use produce_registration_map()'
                            'before trying to display the registration map.')

        fig, ax = plt.subplots(2, 3)
        for i, d in enumerate(['H', 'V', 'D']):
            ax[0, i].imshow(self.registration_map_rel[i], cmap='gray')
            ax[0, i].set_title('Rel reg. map ' + d)
            ax[1, i].imshow(self.registration_map_abs[i], cmap='gray')
            ax[1, i].set_title('Abs reg. map ' + d)

    def dump_stitcher(self, path):
        """
        Use dill to store a tgraph object.

        Parameters
        ----------
        path: (str) path to save the database.

        """
        if path[-4:] != '.pkl':
            path = path + '.pkl'

        with open(path, 'wb') as f:
            dill.dump(self, f)

    def _build_sparse_graphs(self):
        """
        Build the sparse graph from the reliability and (row, col). This method needs to be called after the
        pair-wise registration has been performed for all neighbors pair.

        """

        self.graph_relia_H = csr_matrix((self.relia_H, (self.cgraph_from, self.cgraph_to)),
                                        shape=(self.n_edges, self.n_edges))
        self.graph_relia_V = csr_matrix((self.relia_V, (self.cgraph_from, self.cgraph_to)),
                                        shape=(self.n_edges, self.n_edges))
        self.graph_relia_D = csr_matrix((self.relia_D, (self.cgraph_from, self.cgraph_to)),
                                        shape=(self.n_edges, self.n_edges))

    def _optimize_sparse_graphs(self):
        """
        Optimize the sparse graph by computing the minimum spanning tree for each direction (H, D, V). This
        method needs to be called after the sparse graphs have been built.

        """

        if self.graph_relia_H is None:
            raise TypeError('Error: sparse graph not build yet, please use build_sparse_graph() before trying to'
                            'perform the optimization.')

        for g in ['graph_relia_H', 'graph_relia_V', 'graph_relia_D']:
            graph = getattr(self, g)
            # Minimum spanning tree
            min_tree = minimum_spanning_tree(graph)

            # Get the "true" neighbors
            min_tree = min_tree.tocoo()
            setattr(self, 'min_tree_' + g[-1], min_tree)
            ctree_from = min_tree.row
            setattr(self, 'ctree_from_' + g[-1], ctree_from)

            ctree_to = min_tree.col
            setattr(self, 'ctree_to_' + g[-1], ctree_to)

    def _produce_registration_map(self):
        """
        Produce the registration map where reg_rel_map[d, row, col] (d = H,V,D) is the relative tile
        position in pixel from the expected one. This method needs to be called after the optimization has been done.

        """

        if self.min_tree_H is None:
            raise TypeError('Error: minimum spanning tree not computed yet, please use optimize_sparse_graph()'
                            'before trying to compute the registration map.')

        # Relative registration
        # Initialize relative registration map
        reg_rel_map = np.zeros((3, self.nrow, self.ncol)) # H,V,D

        for i, min_tree in enumerate(['min_tree_H', 'min_tree_V', 'min_tree_D']):
            # Fill it by following the tree and getting the corresponding registration parameters
            # H TREE
            node_array = depth_first_order(getattr(self, min_tree), i_start=0,
                                           directed=False, return_predecessors=False)
            node_visited = [0]

            tree = getattr(self, min_tree)
            row = tree.row
            col = tree.col

            for node_to in zip(node_array[1:]):
                # The previous node in the MST is a visited node with an edge to the current node
                neighbors = []
                for r, c in zip(row, col):
                    if r == node_to:
                        neighbors.append(c)
                    if c == node_to:
                        neighbors.append(r)
                node_from = [x for x in neighbors if x in node_visited]
                node_visited.append(node_to)

                # Get the previous neighbor local reg parameter
                ind1, ind2 = np.unravel_index(node_from, shape=(self.nrow, self.ncol))
                d_neighbor = reg_rel_map[i, ind1, ind2]

                # Get the current 2D tile position
                ind1, ind2 = np.unravel_index(node_to, shape=(self.nrow, self.ncol))
                # Get the associated ind position in the registration graph (as opposed to the reliability min_tree)
                ind_graph = self._get_ind(node_from, node_to)
                # Get the corresponding reg parameter
                d = getattr(self, 'd' + min_tree[-1])[ind_graph]
                # Update the local reg parameter in the 2D matrix
                if node_to > node_from[0]:
                    reg_rel_map[i, ind1, ind2] = d_neighbor + d
                else:
                    reg_rel_map[i, ind1, ind2] = d_neighbor - d
        self.registration_map_rel = reg_rel_map

        reg_abs_map = np.zeros_like(reg_rel_map)
        # H
        for x in range(reg_abs_map.shape[2]):
            reg_abs_map[0, :, x] = reg_rel_map[0, :, x] + x * (self.frame_size-self.overlap)
        # V
        for x in range(reg_abs_map.shape[2]):
            reg_abs_map[1, x, :] = reg_rel_map[1, x, :] + x * (self.frame_size-self.overlap)
        # D
        reg_abs_map[2] = reg_rel_map[2]
        self.registration_map_abs = reg_abs_map

        return reg_rel_map, reg_abs_map

    def _build_database(self):
        """
        Build the database for storing the registration parameters. This method needs to be called after
        the registration map has been produced.

        """

        if self.registration_map_rel is None:
            raise TypeError('Error: database can''t be build if the registration map has not been computed.'
                            ' Please use produce_registration_map() method first.')

        database_dict = {}
        for i in range(self.n_vertex):
            row = self.tiles[i]['row']
            col = self.tiles[i]['col']
            database_dict[i] = {'path': self.tiles[i]['path'],
                                'row': row,
                                'col': col,
                                'dH': self.registration_map_rel[0, row, col],
                                'dV': self.registration_map_rel[1, row, col],
                                'dD': self.registration_map_rel[2, row, col],
                                'ABS_H': self.registration_map_abs[0, row, col],
                                'ABS_V': self.registration_map_abs[1, row, col],
                                'ABS_D': self.registration_map_abs[2, row, col]}

        self.database = pd.DataFrame.from_dict(database_dict, orient='index')

        # Finally set the origin so that tile on the edge have coordinate 0 (rather than negative):
        for i, d in enumerate(['ABS_D', 'ABS_V', 'ABS_H']):
            self.database[d] = self.database[d] - self.database[d].min()

    def _get_ind(self, ind_from, ind_to):
        """
        Returns the ind in the original graph which corresponds to (ind_from, ind_to) in the minimum spanning tree.

        Parameters
        ----------
        ind_from: (int) starting node in the directed graph
        ind_to: (int) ending node in the directed graph

        Returns
        ----------
        (int) corresponding ind in the original graph

        """

        ind = None
        for i, f in enumerate(self.cgraph_from):
            if f == ind_from:
                if self.cgraph_to[i] == ind_to:
                    ind = i
        if ind is None:
            for i, f in enumerate(self.cgraph_to):
                if f == ind_from:
                    if self.cgraph_from[i] == ind_to:
                        ind = i
        if ind is None:
            raise ValueError('Error: can''t find matching vertex pair.')
        return ind

    def _compute_east_registration(self, u, v):
        """
        Compute the registration between the current tile and its eastern neighbor.

        Parameters
        ----------
        u: (list) current tile
        v: (list) neighboring tile

        Returns
        -------
        None

        """

        apr_1, parts_1 = u
        apr_2, parts_2 = v

        patch = pyapr.ReconPatch()
        patch.y_begin = self.frame_size - self.overlap
        proj_zy1, proj_zx1, proj_yx1 = _get_max_proj_apr(apr_1, parts_1, patch, plot=False)

        patch = pyapr.ReconPatch()
        patch.y_end = self.overlap
        proj_zy2, proj_zx2, proj_yx2 = _get_max_proj_apr(apr_2, parts_2, patch, plot=False)

        # if self.row==0 and self.col==1:
        #     proj1, proj2 = [proj_zy1, proj_zx1, proj_yx1], [proj_zy2, proj_zx2, proj_yx2]
        #     for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #         fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #         ax[0].imshow(proj1[i], cmap='gray')
        #         ax[0].set_title('EAST')
        #         ax[1].imshow(proj2[i], cmap='gray')
        #         ax[1].set_title(title)
        #     print('ok')

        if self.mask:
            return _get_masked_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                           [proj_zy2, proj_zx2, proj_yx2],
                                           threshold=self.threshold)
        else:
            return _get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                    [proj_zy2, proj_zx2, proj_yx2])

    def _compute_south_registration(self, u, v):
        """
        Compute the registration between the current tile and its southern neighbor.

        Parameters
        ----------
        u: (list) current tile
        v: (list) neighboring tile

        Returns
        -------
        None

        """
        apr_1, parts_1 = u
        apr_2, parts_2 = v

        patch = pyapr.ReconPatch()
        patch.x_begin = self.frame_size - self.overlap
        proj_zy1, proj_zx1, proj_yx1 = _get_max_proj_apr(apr_1, parts_1, patch, plot=False)

        patch = pyapr.ReconPatch()
        patch.x_end = self.overlap
        proj_zy2, proj_zx2, proj_yx2 = _get_max_proj_apr(apr_2, parts_2, patch, plot=False)

        # if self.row==0 and self.col==1:
        #     print('ok')
        #     proj1, proj2 = [proj_zy1, proj_zx1, proj_yx1], [proj_zy2, proj_zx2, proj_yx2]
        #     for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #         fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #         ax[0].imshow(proj1[i], cmap='gray')
        #         ax[0].set_title('EAST')
        #         ax[1].imshow(proj2[i], cmap='gray')
        #         ax[1].set_title(title)

        if self.mask:
            return _get_masked_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                           [proj_zy2, proj_zx2, proj_yx2],
                                           threshold=self.threshold)
        else:
            return _get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                    [proj_zy2, proj_zx2, proj_yx2])


class channelStitcher(baseStitcher):
    """
    Class used to perform the stitching between different channels. The registration must be performed first a single
    channel (typically auto-fluorescence)

    The stitching is performed between each corresponding tile and the relative displacement is added to the previously
    determined stitching parameters.

    The number and position of tile should matched for bot dataset.
    """

    def __init__(self,
                 stitcher: tileStitcher,
                 tiles_stitched: tileParser,
                 tiles_channel: tileParser):
        """

        Parameters
        ----------
        stitcher: (tileStitcher) tileStitcher object with the multitile registration parameters
        tiles_stitched: (tileParser) tiles corresponding to the stitcher
        tiles_channel: (tileParser) tiles to be registered to tiles_stitched
        """

        super().__init__(tiles_stitched)

        self.stitcher = stitcher
        self.tiles_channel = tiles_channel
        self.database = stitcher.database.copy()

    def compute_rigid_registration(self):

        for t1, t2 in zip(self.tiles, self.tiles_channel):
            tile1 = tileLoader(t1)
            tile1.load_tile()
            apr1, parts1 = tile1.data
            tile2 = tileLoader(t2)
            tile2.load_tile()
            apr2, parts2 = tile2.data

            patch = pyapr.ReconPatch()
            proj1 = _get_max_proj_apr(apr1, parts1, patch)

            patch = pyapr.ReconPatch()
            proj2 = _get_max_proj_apr(apr2, parts2, patch)

            if self.mask:
                d, error = _get_masked_proj_shifts(proj1, proj2, self.threshold)
            else:
                d, error = _get_proj_shifts(proj1, proj2)

            self._update_database(tile2.row, tile2.col, d)

    def _update_database(self, row, col, d):
        d = np.concatenate([d, d])
        df = self.database
        for loc, value in zip(['dD', 'dV', 'dH', 'ABS_D', 'ABS_V', 'ABS_H'], d):
            df.loc[(df['row'] == row) & (df['col'] == col), loc] += value


class tileMerger():
    """
    Class to merge tiles and create a stitched volume. Typically used at a lower resolution for registering
    the sample to an Atlas.

    """
    def __init__(self, tiles, database, n_planes):
        """

        Parameters
        ----------
        tiles:  (tileParser) tileParser object containing the dataset to merge.
        database: (pd.DataFrame, str) database or path to the database containing the registered tile position
        n_planes: (int) number of planes per files.
        """

        if isinstance(database, str):
            self.database = pd.read_csv(database)
        else:
            self.database = database
        self.tiles = tiles
        self.type = tiles.type
        self.frame_size = tiles.frame_size
        self.n_planes = n_planes
        self.n_tiles = tiles.n_tiles
        self.n_row = tiles.nrow
        self.n_col = tiles.ncol

        # Size of the merged array (to be defined when the merged array is initialized).
        self.nx = None
        self.ny = None
        self.nz = None

        self.downsample = 1
        self.level_delta = 0
        self.merged_data = None

    def merge_additive(self, mode='constant'):
        """
        Perform merging with an additive algorithm for overlapping area. Maximum merging should be prefered to
        avoid integer overflowing and higher signals and the overlapping areas.

        Parameters
        ----------
        mode: (str) APR reconstruction type among ('constant', 'smooth', 'level')

        Returns
        -------
        None

        """
        H_pos = self.database['ABS_H'].to_numpy()
        H_pos = (H_pos - H_pos.min())/self.downsample
        V_pos = self.database['ABS_V'].to_numpy()
        V_pos = (V_pos - V_pos.min())/self.downsample
        D_pos = self.database['ABS_D'].to_numpy()
        D_pos = (D_pos - D_pos.min())/self.downsample

        for i in range(self.n_tiles):
            apr, parts = self._load_tile(i)
            u = pyapr.data_containers.APRSlicer(apr, parts, level_delta=self.level_delta, mode=mode)
            data = u[:, :, :]

            x1 = int(H_pos[i])
            x2 = int(H_pos[i] + data.shape[2])
            y1 = int(V_pos[i])
            y2 = int(V_pos[i] + data.shape[1])
            z1 = int(D_pos[i])
            z2 = int(D_pos[i] + data.shape[0])

            self.merged_data[z1:z2, y1:y2, x1:x2] = self.merged_data[z1:z2, y1:y2, x1:x2] + data
            self.merged_data = self.merged_data.astype('uint16')

    def merge_max(self, mode='constant'):
        """
        Perform merging with a maximum algorithm for overlapping area.

        Parameters
        ----------
        mode: (str) APR reconstruction type among ('constant', 'smooth', 'level')

        Returns
        -------
        None

        """
        H_pos = self.database['ABS_H'].to_numpy()
        H_pos = (H_pos - H_pos.min())/self.downsample
        V_pos = self.database['ABS_V'].to_numpy()
        V_pos = (V_pos - V_pos.min())/self.downsample
        D_pos = self.database['ABS_D'].to_numpy()
        D_pos = (D_pos - D_pos.min())/self.downsample

        with alive_bar(total=self.n_tiles, title='Merging', force_tty=True) as bar:
            for i, t in enumerate(self.tiles):
                tile = tileLoader(t)
                tile.load_tile()
                apr, parts = tile.data
                u = pyapr.data_containers.APRSlicer(apr, parts, level_delta=self.level_delta, mode=mode)
                data = u[:, :, :]

                x1 = int(H_pos[i])
                x2 = int(H_pos[i] + data.shape[2])
                y1 = int(V_pos[i])
                y2 = int(V_pos[i] + data.shape[1])
                z1 = int(D_pos[i])
                z2 = int(D_pos[i] + data.shape[0])

                self.merged_data[z1:z2, y1:y2, x1:x2] = np.maximum(self.merged_data[z1:z2, y1:y2, x1:x2], data)
                bar()
        self.merged_data = self.merged_data.astype('uint16')

    def crop(self, background=0, xlim=None, ylim=None, zlim=None):
        """
        Add a black mask around the brain (rather than really cropping which makes the overlays complicated in
        a later stage).

        Parameters
        ----------
        background: (int) constant value to replace the cropped area with.
        xlim: (list) x limits for cropping
        ylim: (list) y limits for cropping
        zlim: (list) z limits for cropping

        Returns
        -------
        None
        """

        if self.merged_data is None:
            raise TypeError('Error: please merge data before cropping.')

        if xlim is not None:
            if xlim[0] != 0:
                self.merged_data[:, :, :xlim[0]] = background
            if xlim[1] != self.merged_data.shape[2]:
                self.merged_data[:, :, xlim[1]:] = background
        if ylim is not None:
            if ylim[0] != 0:
                self.merged_data[:, :ylim[0], :] = background
            if ylim[1] != self.merged_data.shape[1]:
                self.merged_data[:, ylim[1]:, :] = background
        if zlim is not None:
            if zlim[0] != 0:
                self.merged_data[:zlim[0], :, :] = background
            if zlim[1] != self.merged_data.shape[0]:
                self.merged_data[zlim[1]:, :, :] = background

    def equalize_hist(self, method='skimage'):
        """
        Perform histogram equalization to improve the contrast on merged data.
        Both OpenCV (only 2D) and Skimage (3D but 10 times slower) are available.

        Parameters
        ----------
        method: (str) method for performing histogram equalization among 'skimage' and 'opencv'.

        Returns
        -------
        None
        """

        if self.merged_data is None:
            raise TypeError('Error: please merge data before equalizing histogram.')

        if method == 'opencv':
            # clahe = cv.createCLAHE(tileGridSize=(8, 8))
            # for i in range(self.merged_data.shape[0]):
                # self.merged_data[i] = clahe.apply(self.merged_data[i])
            print('opencv not currently supported due to incompatibility with pyqt5')
        elif method == 'skimage':
            self.merged_data = equalize_adapthist(self.merged_data)
        else:
            raise ValueError('Error: unknown method for adaptive histogram normalization.')

    def initialize_merged_array(self):
        """
        Initialize the merged array in accordance with the asked downsampling.

        Returns
        -------
        None
        """

        self.nx = int(np.ceil(self._get_nx() / self.downsample))
        self.ny = int(np.ceil(self._get_ny() / self.downsample))
        self.nz = int(np.ceil(self._get_nz() / self.downsample))

        self.merged_data = np.zeros((self.nz, self.ny, self.nx))

    def set_downsample(self, downsample):
        """
        Set the downsampling value for the merging reconstruction.

        Parameters
        ----------
        downsample: (int) downsample factor

        Returns
        -------
        None

        """

        # TODO: find a more rigorous way of enforcing this. (Probably requires that the APR is loaded).
        if downsample not in [1, 2, 4, 8, 16, 32]:
            raise ValueError('Error: downsample value should be compatible with APR levels.')

        self.downsample = downsample
        self.level_delta = int(-np.log2(self.downsample))

    def _get_nx(self):
        """
        Compute the merged array size for x dimension.

        Returns
        -------
        (int) x size for merged array
        """
        x_pos = self.database['ABS_H'].to_numpy()
        return x_pos.max() - x_pos.min() + self.frame_size

    def _get_ny(self):
        """
        Compute the merged array size for y dimension.

        Returns
        -------
        (int) y size for merged array
        """
        y_pos = self.database['ABS_V'].to_numpy()
        return y_pos.max() - y_pos.min() + self.frame_size

    def _get_nz(self):
        """
        Compute the merged array size for y dimension.

        Returns
        -------
        (int) y size for merged array
        """
        z_pos = self.database['ABS_D'].to_numpy()
        return z_pos.max() - z_pos.min() + self.n_planes