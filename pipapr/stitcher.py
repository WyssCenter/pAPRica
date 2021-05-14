"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from glob import glob
import os
from skimage.io import imread
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

class tileStitcher():
    def __init__(self,
                 tiles: tileParser):
        
        self.tiles = tiles
        self.ncol = tiles.ncol
        self.nrow = tiles.nrow
        self.n_vertex = tiles.n_tiles
        self.n_edges = tiles.n_edges
        self.overlap = tiles.overlap
        self.frame_size = tiles.frame_size
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
        
        self.mask = False
        self.threshold = None

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

    def compute_registration(self):
        """
        Compute the pair-wise registration for a given tile with all its neighbors (EAST and SOUTH to avoid
        the redundancy).

        Parameters
        ----------
        tgraph: (tileGraph object) stores pair-wise registration to further perform the global optimization.

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
    
    def build_sparse_graphs(self):
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

    def optimize_sparse_graphs(self):
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

    def plot_graph(self, annotate=False):
        """
        Plot the graph for each direction (H, D, V). This method needs to be called after the graph
        optimization.

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

    def produce_registration_map(self):
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

    def build_database(self, tiles):
        """
        Build the database for storing the registration parameters. This method needs to be called after
        the registration map has been produced.

        """

        if self.registration_map_rel is None:
            raise TypeError('Error: database can''t be build if the registration map has not been computed.'
                            ' Please use produce_registration_map() method first.')
        self.database = pd.DataFrame(columns=['path',
                                            'row',
                                            'col',
                                            'dH',
                                            'dV',
                                            'dD',
                                            'ABS_H',
                                            'ABS_V',
                                            'ABS_D'])
        for i in range(self.n_vertex):
            row = tiles[i]['row']
            col = tiles[i]['col']
            self.database.loc[i] = [tiles[i]['path'], row, col,
                                    self.registration_map_rel[0, row, col],
                                    self.registration_map_rel[1, row, col],
                                    self.registration_map_rel[2, row, col],
                                    self.registration_map_abs[0, row, col],
                                    self.registration_map_abs[1, row, col],
                                    self.registration_map_abs[2, row, col]]

    def save_database(self, path):
        """
        Save database at the given path. The database must be built before calling this method.

        """

        if self.database is None:
            raise TypeError('Error: database can''t be saved because it was not created. '
                            'Please call build_database() first.')

        self.database.to_csv(path)

    def dump_stitcher(self, path):
        """
        Use dill to store a tgraph object.

        """
        if path[-4:] != '.pkl':
            path = path + '.pkl'

        with open(path, 'wb') as f:
            dill.dump(self, f)

    def _get_ind(self, ind_from, ind_to):
        """
        Returns the ind in the original graph which corresponds to (ind_from, ind_to) in the minimum spanning tree.
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

    def _get_proj_shifts(self, proj1, proj2, upsample_factor=1):
        """
        This function computes shifts from max-projections on overlapping areas. It uses the phase cross-correlation
        to compute the shifts.

        Parameters
        ----------
        proj1: (list of arrays) max-projections for tile 1
        proj2: (list of arrays) max-projections for tile 2

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

        # for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('dx={}, dy={}, dz={}'.format(dx, dy, dz))
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)
        #
        # if self.row==0 and self.col==0:
        #     print('ok')

        return np.array([dz, dy, dx]), np.array([rz, ry, rx])

    def _get_masked_proj_shifts(self, proj1, proj2, upsample_factor=1):
        """
        This function computes shifts from max-projections on overlapping areas with mask on brightest area.
        It uses the phase cross-correlation to compute the shifts.

        Parameters
        ----------
        proj1: (list of arrays) max-projections for tile 1
        proj2: (list of arrays) max-projections for tile 2

        Returns
        -------
        shifts in (x, y, z) and error measure (0=reliable, 1=not reliable)

        """
        # Compute mask to discard very bright area that are likely bubbles or artefacts
        mask_ref = []
        mask_move = []
        for i in range(3):
            vmax = np.percentile(proj1[i], self.threshold)
            mask_ref.append(proj1[i] < vmax)
            vmax = np.percentile(proj2[i], self.threshold)
            mask_move.append(proj2[i] < vmax)

        # Compute phase cross-correlation to extract shifts
        dzy = phase_cross_correlation(proj1[0], proj2[0],
                                      return_error=True, upsample_factor=upsample_factor,
                                      reference_mask=mask_ref[0], moving_mask=mask_move[0])
        error_zy = self._get_registration_error(proj1[0], proj2[0])
        dzx = phase_cross_correlation(proj1[1], proj2[1],
                                      return_error=True, upsample_factor=upsample_factor,
                                      reference_mask=mask_ref[1], moving_mask=mask_move[1])
        error_zx = self._get_registration_error(proj1[1], proj2[1])
        dyx = phase_cross_correlation(proj1[2], proj2[2],
                                      return_error=True, upsample_factor=upsample_factor,
                                      reference_mask=mask_ref[2], moving_mask=mask_move[2])
        error_yx = self._get_registration_error(proj1[2], proj2[2])

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

        # for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('dx={}, dy={}, dz={}'.format(dx, dy, dz))
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)
        #
        # if self.row==0 and self.col==0:
        #     print('ok')

        return np.array([dz, dy, dx]), np.array([rz, ry, rx])

    def _get_registration_error(self, proj1, proj2):
        return np.sqrt(1-correlate(proj1, proj2).max()**2/(np.sum(proj1**2)*np.sum(proj2**2)))

    def _get_max_proj_apr(self, apr, parts, patch, plot=False):
        """
        Get the maximum projection from 3D APR data.
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

    def _compute_east_registration(self, u, v):
        """
        Compute the registration between the current tile and its eastern neighbor.
        """
        apr_1, parts_1 = u
        apr_2, parts_2 = v

        patch = pyapr.ReconPatch()
        patch.y_begin = self.frame_size - self.overlap
        proj_zy1, proj_zx1, proj_yx1 = self._get_max_proj_apr(apr_1, parts_1, patch, plot=False)

        patch = pyapr.ReconPatch()
        patch.y_end = self.overlap
        proj_zy2, proj_zx2, proj_yx2 = self._get_max_proj_apr(apr_2, parts_2, patch, plot=False)

        # proj1, proj2 = [proj_zy1, proj_zx1, proj_yx1], [proj_zy2, proj_zx2, proj_yx2]
        # for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('EAST')
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)

        # if self.row==0 and self.col==1:
        #     print('ok')

        if self.mask:
            return self._get_masked_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                         [proj_zy2, proj_zx2, proj_yx2])
        else:
            return self._get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                         [proj_zy2, proj_zx2, proj_yx2])

    def _compute_south_registration(self, u, v):
        """
        Compute the registration between the current tile and its southern neighbor.
        """
        apr_1, parts_1 = u
        apr_2, parts_2 = v

        patch = pyapr.ReconPatch()
        patch.x_begin = self.frame_size - self.overlap
        proj_zy1, proj_zx1, proj_yx1 = self._get_max_proj_apr(apr_1, parts_1, patch, plot=False)

        patch = pyapr.ReconPatch()
        patch.x_end = self.overlap
        proj_zy2, proj_zx2, proj_yx2 = self._get_max_proj_apr(apr_2, parts_2, patch, plot=False)

        # proj1, proj2 = [proj_zy1, proj_zx1, proj_yx1], [proj_zy2, proj_zx2, proj_yx2]
        # for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('EAST')
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)

        # if self.row==0 and self.col==1:
        #     print('ok')

        if self.mask:
            return self._get_masked_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                         [proj_zy2, proj_zx2, proj_yx2])
        else:
            return self._get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                         [proj_zy2, proj_zx2, proj_yx2])


class tileMerger():
    def __init__(self, database, frame_size, n_planes, type):

        if isinstance(database, str):
            self.database = pd.read_csv(database)
        else:
            self.database = database
        self.type = type
        self.frame_size = frame_size
        self.n_planes = n_planes
        self.n_tiles = len(self.database)
        self.n_row = self.database['row'].max()-self.database['row'].min()+1
        self.n_col = self.database['col'].max()-self.database['col'].min()+1

        # Size of the merged array (to be defined when the merged array is initialized).
        self.nx = None
        self.ny = None
        self.nz = None

        self.downsample = 1
        self.level_delta = 0
        self.merged_data = None

    def merge_additive(self, mode='constant'):
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
        H_pos = self.database['ABS_H'].to_numpy()
        H_pos = (H_pos - H_pos.min())/self.downsample
        V_pos = self.database['ABS_V'].to_numpy()
        V_pos = (V_pos - V_pos.min())/self.downsample
        D_pos = self.database['ABS_D'].to_numpy()
        D_pos = (D_pos - D_pos.min())/self.downsample

        with alive_bar(total=self.n_tiles, title='Merging', force_tty=True) as bar:
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

                self.merged_data[z1:z2, y1:y2, x1:x2] = np.maximum(self.merged_data[z1:z2, y1:y2, x1:x2], data)
                bar()
        self.merged_data = self.merged_data.astype('uint16')

    def crop(self, background=0, xlim=None, ylim=None, zlim=None):
        """
        Add a black mask around the brain (rather than really cropping which makes the overlays complicated in
        a later stage).

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

    def _load_tile(self, i):
        """
        Load the current tile.
        """
        path = self.database['path'].loc[i]
        if self.type == 'tiff2D':
            files = glob(os.path.join(path, '*.tif'))
            im = imread(files[0])
            u = np.zeros((len(files), *im.shape))
            u[0] = im
            files.pop(0)
            for i, file in enumerate(files):
                u[i+1] = imread(file)
        elif self.type == 'tiff3D':
            u = imread(path)
        elif self.type == 'apr':
            apr = pyapr.APR()
            parts = pyapr.ShortParticles()
            pyapr.io.read(path, apr, parts)
            u = (apr, parts)
        else:
            raise TypeError('Error: image type {} not supported.'.format(self.type))
        return u

    def initialize_merged_array(self):
        """
        Initialize the merged array in accordance with the asked downsampling.

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

        """

        # TODO: find a more rigorous way of enforcing this. (Probably requires that the APR is loaded).
        if downsample not in [1, 2, 4, 8, 16, 32]:
            raise ValueError('Error: downsample value should be compatible with APR levels.')

        self.downsample = downsample
        self.level_delta = int(-np.log2(self.downsample))

    def _get_nx(self):
        x_pos = self.database['ABS_H'].to_numpy()
        return x_pos.max() - x_pos.min() + self.frame_size

    def _get_ny(self):
        y_pos = self.database['ABS_V'].to_numpy()
        return y_pos.max() - y_pos.min() + self.frame_size

    def _get_nz(self):
        z_pos = self.database['ABS_D'].to_numpy()
        return z_pos.max() - z_pos.min() + self.n_planes


class atlas():

    def __init__(self, path, x_downsample, z_downsample, y_downsample=None):
        self.x_downsample = x_downsample
        if y_downsample is None:
            self.y_downsample = x_downsample
        else:
            self.y_downsample = y_downsample
        self.z_downsample = z_downsample
        self.atlas = imread(path)

    def get_loc_id(self, x, y, z):
        return self.atlas[int(z/self.z_downsample), int(y/self.y_downsample), int(x/self.x_downsample)]