from glob import glob
import os
from skimage.io import imread
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import pandas as pd
import pyapr
from skimage.registration import phase_cross_correlation
import re


class tileParser():
    def __init__(self, path):
        self.path = path
        self.tiles_list = self.get_tile_list()
        self.n_tiles = len(self.tiles_list)
        self.ncol = self.get_ncol()
        self.nrow = self.get_nrow()
        self.neighbors, self.n_edges = self.get_neighbors_map()
        self.path_list = self.get_path_list()
        self.overlap = self.get_overlap()

    def get_overlap(self):
        u = imread(glob(os.path.join(self.tiles_list[0]['path'], '*.tif'))[0])
        nx = u.shape[1]

        tile1 = self.tiles_list[0]['path']
        tile2 = self.tiles_list[1]['path']

        str1 = re.findall(r'(\d{6})_(\d{6})', tile1)[0]
        str2 = re.findall(r'(\d{6})_(\d{6})', tile2)[0]

        if int(str1[0]) - int(str2[0]) != 0:
            overlap = nx - np.abs((int(str1[0]) - int(str2[0]))/10)
        elif int(str1[1]) - int(str2[1]) != 0:
            overlap = nx - np.abs((int(str1[1]) - int(str2[1]))/10)
        else:
            raise ValueError('Error: can''t infer overlap.')

        return int(overlap)

    def get_tile_list(self):
        H_folders = [f.path for f in os.scandir(self.path) if f.is_dir()]
        tiles = []
        for i, H_path in enumerate(H_folders):
            V_folders = [f.path for f in os.scandir(H_path) if f.is_dir()]
            for ii, v_path in enumerate(V_folders):
                tile = {'path': v_path,
                        'row': i,
                        'col': ii,
                        'type': self.get_type(v_path)}
                tiles.append(tile)
        return tiles

    def get_type(self, path):
        """
        Return the type of image files either 'tiff2d' of 'tiff3d'
        """
        n_files = len(glob(os.path.join(path, '*.tif')))
        if n_files > 1:
            return 'tiff2D'
        elif n_files == 1:
            return 'tiff3D'
        else:
            raise TypeError('Error: no tiff files found in {}.'.format(path))

    def get_ncol(self):
        ncol = 0
        for tile in self.tiles_list:
            if tile['col']>ncol:
                ncol = tile['col']
        return ncol+1

    def get_nrow(self):
        nrow = 0
        for tile in self.tiles_list:
            if tile['row']>nrow:
                nrow = tile['row']
        return nrow+1

    def get_total_neighbors_map(self):
        """
        Return the total neighbors maps (with redundancy in the cas of undirected graph)
        """
        # Initialize neighbors
        neighbors = [None] * self.ncol
        for x in range(self.ncol):
            # Create new dimension
            neighbors[x] = [None] * self.nrow
            for y in range(self.nrow):
                # Fill up 2D list
                tmp = []
                if x > 0:
                    # NORTH
                    tmp.append([x-1, y])
                if x < self.ncol-1:
                    # SOUTH
                    tmp.append([x+1, y])
                if y > 0:
                    # WEST
                    tmp.append([x, y-1])
                if y < self.nrow-1:
                    # EAST
                    tmp.append([x, y+1])
                neighbors[x][y] = tmp
        return neighbors

    def get_neighbors_map(self):
        """
        Returns the non-redundant neighbors map: neighbors[row][col] gives a list of neighbors. Only
        SOUTH and EAST are returned to avoid the redundancy.
        """
        # Initialize neighbors
        neighbors = [None] * self.ncol
        cnt = 0
        for x in range(self.ncol):
            # Create new dimension
            neighbors[x] = [None] * self.nrow
            for y in range(self.nrow):
                # Fill up 2D list
                tmp = []
                if x < self.ncol-1:
                    # SOUTH
                    tmp.append([x+1, y])
                    cnt += 1
                if y < self.nrow-1:
                    # EAST
                    tmp.append([x, y+1])
                    cnt += 1
                neighbors[x][y] = tmp
        return neighbors, cnt

    def get_path_list(self):
        path_list = []
        for tile in self.tiles_list:
            path_list.append(tile['path'])
        return path_list

    def __getitem__(self, item):
        e = self.tiles_list[item]
        e['neighbors'] = self.neighbors[e['row']][e['col']]
        neighbors_path = []
        for x, y in e['neighbors']:
            neighbors_path.append(self.path_list[y + x * self.nrow])
        e['neighbors_path'] = neighbors_path
        e['overlap'] = self.overlap
        return e

    def __len__(self):
        return self.n_tiles


class tileLoader():
    """
    Load tile data and neighboring tiles.
    """
    def __init__(self, tile):
        self.path = tile['path']
        self.row = tile['row']
        self.col = tile['col']
        self.type = tile['type']
        self.neighbors = tile['neighbors']
        self.neighbors_path = tile['neighbors_path']
        self.overlap = tile['overlap']
        self.data = self.load_tile(self.path)
        self.data_neighbors = self.load_neighbors()
        self.convert_to_apr()

    def convert_to_apr(self):
        # Parameters are hardcoded for now
        par = pyapr.APRParameters()
        par.auto_parameters = False  # really heuristic and not working
        par.sigma_th = 26.0
        par.grad_th = 3.0
        par.Ip_th = 253.0
        par.rel_error = 0.2
        par.gradient_smoothing = 2

        # Convert data to APR
        self.data = pyapr.converter.get_apr(image=self.data, params=par, verbose=False)

        # Convert neighbors to APR
        data_apr = []
        for data in self.data_neighbors:
            data_apr.append(pyapr.converter.get_apr(image=data, params=par, verbose=False))
        self.data_neighbors = data_apr

    def load_tile(self, path):
        if self.type == 'tiff2D':
            files = glob(os.path.join(path, '*.tif'))
            im = imread(files[0])
            u = np.zeros((len(files), *im.shape))
            u[0] = im
            files.pop(0)
            for i, file in enumerate(files):
                u[i+1] = imread(file)
        elif self.type == 'tiff3D':
            u = imread(*glob(os.path.join(path, '*.tif')))
        else:
            raise TypeError('Error: image type {} not supported.'.format(self.type))
        return u

    def load_neighbors(self):
        u = []
        for neighbor in self.neighbors_path:
            u.append(self.load_tile(neighbor))
        return u

    def compute_registration(self, tgraph):
        for v, coords in zip(self.data_neighbors, self.neighbors):
            if self.row==coords[0] and self.col<coords[1]:
                # EAST
                reg, rel = self._compute_east_registration(v)

            elif self.col==coords[1] and self.row<coords[0]:
                # SOUTH
                reg, rel = self._compute_south_registration(v)

            else:
                raise TypeError('Error: couldn''t determine registration to perform.')

            tgraph.cgraph_from.append(np.ravel_multi_index([self.row, self.col], dims=(tgraph.nrow, tgraph.ncol)))
            tgraph.cgraph_to.append(np.ravel_multi_index([coords[0], coords[1]], dims=(tgraph.nrow, tgraph.ncol)))
            tgraph.dH.append(reg[1])
            tgraph.dV.append(reg[2])
            tgraph.dD.append(reg[0])
            tgraph.relia_H.append(rel[1])
            tgraph.relia_V.append(rel[2])
            tgraph.relia_D.append(rel[0])

    def _get_max_proj_apr(self, apr, parts, plot=False):
        proj = []
        for d in range(3):
            # dim=0: project along Y to produce a ZX plane
            # dim=1: project along X to produce a ZY plane
            # dim=2: project along Z to produce an XY plane
            proj.append(pyapr.numerics.transform.maximum_projection(apr, parts, dim=d))

        if plot:
            fig, ax = plt.subplots(1, 3)
            for i in range(3):
                ax[i].imshow(proj[i], cmap='gray')
        return proj[0], proj[1], proj[2]

    def _get_proj_shifts(self, proj1, proj2, upsample_factor=10):
        """
        This function computes shifts from max-projections on overlapping areas. It uses the phase cross-correlation
        to compute the shifts.

        Parameters
        ----------
        proj1: (list of arrays) max-projections for tile 1
        proj2: (list of arrays) max-projections for tile 2

        Returns
        -------
        shifts in (x, y, z) and relialability measure (0=lowest, 1=highest)
        """
        # Compute phase cross-correlation to extract shifts
        dzx, error_zx, _ = phase_cross_correlation(proj1[0], proj2[0],
                                                   return_error=True, upsample_factor=upsample_factor)
        dzy, error_zy, _ = phase_cross_correlation(proj1[1], proj2[1],
                                                   return_error=True, upsample_factor=upsample_factor)
        dxy, error_xy, _ = phase_cross_correlation(proj1[2], proj2[2],
                                                   return_error=True, upsample_factor=upsample_factor)

        # Keep only the most reliable registration
        if error_zx < error_zy:
            dz = dzx[0]
            rz = error_zx
        else:
            dz = dzy[0]
            rz = error_zy
        if error_zx < error_xy:
            dx = dzx[1]
            rx = error_zx
        else:
            dx = dxy[0]
            rx = error_xy
        if error_xy < error_zy:
            dy = dxy[1]
            ry = error_xy
        else:
            dy = dzy[1]
            ry = error_zy

        return np.array([dz, dx, dy]), np.array([rz, rx, ry])

    def _compute_east_registration(self, v):
        apr_1, parts_1 = self.data
        apr_2, parts_2 = v

        # TODO: use the crop in maxproj directly in APR for improving the speed
        proj_xz1, proj_zy1, proj_xy1 = self._get_max_proj_apr(apr_1, parts_1, plot=False)
        proj_xz1 = proj_xz1[:, -self.overlap:]
        proj_xy1 = proj_xy1[:, -self.overlap:]

        proj_xz2, proj_zy2, proj_xy2 = self._get_max_proj_apr(apr_2, parts_2, plot=False)
        proj_xz2 = proj_xz2[:, :self.overlap]
        proj_xy2 = proj_xy2[:, :self.overlap]

        return self._get_proj_shifts([proj_xz1, proj_zy1, proj_xy1],
                                     [proj_xz2, proj_zy2, proj_xy2])

    def _compute_south_registration(self, v):
        apr_1, parts_1 = self.data
        apr_2, parts_2 = v

        # TODO: use the crop in maxproj directly in APR for improving the speed
        proj_xz1, proj_zy1, proj_xy1 = self._get_max_proj_apr(apr_1, parts_1, plot=False)
        proj_zy1 = proj_zy1[-self.overlap:, :]
        proj_xy1 = proj_xy1[-self.overlap:, :]

        proj_xz2, proj_zy2, proj_xy2 = self._get_max_proj_apr(apr_2, parts_2, plot=False)
        proj_zy2 = proj_zy2[:self.overlap, :]
        proj_xy2 = proj_xy2[:self.overlap, :]

        return self._get_proj_shifts([proj_xz1, proj_zy1, proj_xy1],
                                         [proj_xz2, proj_zy2, proj_xy2])


class tileGraph():
    """
    Class object for the graph (sparse matrix) to be build up and optimized.

    To be initialized with a tileParser object.

    """
    def __init__(self, tiles):
        self.ncol = tiles.ncol
        self.nrow = tiles.nrow
        self.n_vertex = self.ncol*self.nrow
        self.n_edges = tiles.n_edges
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

    def plot_min_trees(self):
        """
        Plot the minimum spanning tree for each direction (H, D, V). This method needs to be called after the graph
        optimization.

        """

        if self.min_tree_H is None:
            raise TypeError('Error: minimum spanning tree not computed yet, please use optimize_sparse_graph()'
                            'before trying to plot the trees.')

        fig, ax = plt.subplots(1,3)
        for i, d in enumerate(['H', 'V', 'D']):
            ind_from = getattr(self, 'ctree_from_' + d)
            row, col = np.unravel_index(ind_from, shape=(self.nrow, self.ncol))
            V1 = np.vstack((row, col)).T

            ind_to = getattr(self, 'ctree_to_' + d)
            row, col = np.unravel_index(ind_to, shape=(self.nrow, self.ncol))
            V2 = np.vstack((row, col)).T

            for ii in range(V1.shape[0]):
                ax[i].plot([V1[ii, 1], V2[ii, 1]], [V1[ii, 0], V2[ii, 0]], 'ko-', markerfacecolor='r')
            ax[i].set_title(d + ' tree')
            ax[i].invert_yaxis()

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
                reg_rel_map[i, ind1, ind2] = d_neighbor + d
        self.registration_map_rel = reg_rel_map
        return reg_rel_map

    def build_database(self, tiles):
        """
        Build the database for storing the registration parameters. This method needs to be called after
        the registration map has been produced.

        """

        if self.registration_map_rel is None:
            raise TypeError('Error: database can''t be build if the registration map has not been computed.'
                            ' Please use produce_registration_map() method first.')
        self.database = pd.DataFrame(columns = ['path',
                                                'row',
                                                'col',
                                                'dH',
                                                'dV',
                                                'dD'])
        for i in range(self.n_vertex):
            row = tiles[i]['row']
            col = tiles[i]['col']
            self.database.loc[i] = [tiles[i]['path'], row, col,
                                    self.registration_map_rel[0, row, col], self.registration_map_rel[1, row, col],
                                    self.registration_map_rel[2, row, col]]

    def save_database(self, path):
        """
        Save database at the given path. The database must be built before calling this method.

        """

        if self.database is None:
            raise TypeError('Error: database can''t be saved because it was not created. '
                            'Please call build_database() first.')

        self.database.to_csv(path)

    def _get_ind(self, ind_from, ind_to):
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


if __name__=='__main__':
    from time import time

    np.random.seed(0)
    t = time()
    tiles = tileParser(r'/media/sf_shared_folder_virtualbox/terastitcher_tests')
    print('Elapsed time parse data: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    tgraph = tileGraph(tiles)
    print('Elapsed time init tgraph: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    for tile in tiles:
        loaded_tile = tileLoader(tile)
        loaded_tile.compute_registration(tgraph)
    print('Elapsed time load and compute pairwise reg: {:.2f} s.'.format(time() - t))

    t = time()
    tgraph.build_sparse_graphs()
    print('Elapsed time build sparse graph: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    tgraph.optimize_sparse_graphs()
    print('Elapsed time optimize graph: {:.2f} ms.'.format((time() - t)*1000))
    tgraph.plot_min_trees()
    t = time()
    reg_map = tgraph.produce_registration_map()
    print('Elapsed time reg map: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    tgraph.build_database(tiles)
    print('Elapsed time build database: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    tgraph.save_database(r'/media/sf_shared_folder_virtualbox/terastitcher_tests/registration_results.csv')
    print('Elapsed time save database: {:.2f} ms.'.format((time() - t)*1000))

    fig, ax = plt.subplots(1, 3)
    for i, d in enumerate(['H', 'V', 'D']):
        ax[i].imshow(reg_map[i], cmap='gray')
        ax[i].set_title('Reg. map ' + d)