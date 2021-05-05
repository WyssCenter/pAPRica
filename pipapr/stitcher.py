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
import matplotlib.pyplot as plt
import pandas as pd
import pyapr
import cv2 as cv
from skimage.exposure import equalize_adapthist
from alive_progress import alive_bar
import dill


class tileGraph():
    """
    Class object for the graph (sparse matrix) to be build up and optimized.

    To be initialized with a tileParser object.

    """
    def __init__(self, tiles):
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

    def dump_tgraph(self, path):
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

    def equalize_hist(self, method='opencv'):
        """
        Perform histogram equalization to improve the contrast on merged data.
        Both OpenCV (only 2D) and Skimage (3D but 10 times slower) are available.
        """

        if self.merged_data is None:
            raise TypeError('Error: please merge data before equalizing histogram.')

        if method == 'opencv':
            clahe = cv.createCLAHE(tileGridSize=(8, 8))
            for i in range(self.merged_data.shape[0]):
                self.merged_data[i] = clahe.apply(self.merged_data[i])
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