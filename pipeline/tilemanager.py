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
from viewer.pyapr_napari import display_layers, apr_to_napari_Image, apr_to_napari_Labels
from joblib import load


class tileParser():
    """
    Class to handle the data in the same fashion as TeraStitcher, see here:
    https://github.com/abria/TeraStitcher/wiki/Supported-volume-formats#two-level-hierarchy-of-folders
    """
    def __init__(self, path):
        self.path = path
        self.tiles_list = self._get_tile_list()
        self.type = self._get_type()
        self.n_tiles = len(self.tiles_list)
        self.ncol = self._get_ncol()
        self.nrow = self._get_nrow()
        self.neighbors, self.n_edges = self._get_neighbors_map()
        self.path_list = self._get_path_list()
        self.overlap, self.frame_size = self._get_overlap()

    def _get_overlap(self):
        """
        Infer the overlap between each tile. This is inferred from the folder names and the frame size.
        The overlap in H and V are supposed to be the same.
        """
        if self.type == 'apr':
            apr = pyapr.APR()
            parts = pyapr.ShortParticles()
            pyapr.io.read(glob(os.path.join(self.tiles_list[0]['path'], '*.apr'))[0], apr, parts)
            nx = apr.x_num(apr.level_max())
        else:
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

        return int(overlap), int(nx)

    def _get_tile_list(self):
        """
        Returns a list of tiles as a dictionary
        """
        H_folders = [f.path for f in os.scandir(self.path) if f.is_dir()]
        tiles = []
        for i, H_path in enumerate(H_folders):
            V_folders = [f.path for f in os.scandir(H_path) if f.is_dir()]
            for ii, v_path in enumerate(V_folders):
                tile = {'path': v_path,
                        'row': i,
                        'col': ii,
                        }
                tiles.append(tile)
        return tiles

    def _get_type(self):
        """
        Return the type of image files either 'tiff2d', 'tiff3d or 'apr'
        The type is inferred from the first tile and all tiles are expected to be of the same type.
        """
        path = self.tiles_list[0]['path']

        # If the number of tiff in each folder is >1 then the type is 'tiff2d' else it's 'tiff3d'
        # If no tiff files then the type is APR.
        # TODO: clean this up because it is not robust if new files are added in folders.
        n_files = len(glob(os.path.join(path, '*.tif')))
        if n_files > 1:
            return 'tiff2D'
        elif n_files == 1:
            return 'tiff3D'
        elif n_files == 0:
            n_files = len(glob(os.path.join(path, '*.apr')))
            if n_files > 0:
                return 'apr'
        else:
            raise TypeError('Error: no tiff files found in {}.'.format(path))

    def _get_ncol(self):
        """
        Returns the number of columns (H) to be stitched.
        """
        ncol = 0
        for tile in self.tiles_list:
            if tile['col'] > ncol:
                ncol = tile['col']
        return ncol+1

    def _get_nrow(self):
        """
        Returns the number of rows (V) to be stitched.
        """
        nrow = 0
        for tile in self.tiles_list:
            if tile['row'] > nrow:
                nrow = tile['row']
        return nrow+1

    def _get_total_neighbors_map(self):
        """
        Return the total neighbors maps (with redundancy in the case of undirected graph).

        Note: this function is not used anymore.
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

    def _get_neighbors_map(self):
        """
        Returns the non-redundant neighbors map: neighbors[row][col] gives a list of neighbors and the total
        number of pair-wise neighbors. Only SOUTH and EAST are returned to avoid the redundancy.
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

    def _get_path_list(self):
        """
        Returns a list containing the path to each tile.
        """
        path_list = []
        for tile in self.tiles_list:
            path_list.append(tile['path'])
        return path_list

    def __getitem__(self, item):
        """
        Return tiles, add neighbors information before returning.
        """
        e = self.tiles_list[item]
        e['neighbors'] = self.neighbors[e['row']][e['col']]
        neighbors_path = []
        for x, y in e['neighbors']:
            neighbors_path.append(self.path_list[y + x * self.nrow])
        e['neighbors_path'] = neighbors_path
        e['type'] = self.type
        e['overlap'] = self.overlap
        e['frame_size'] = self.frame_size
        return e

    def __len__(self):
        """
        Returns the number of tiles.
        """
        return self.n_tiles


class tileLoader():
    """
    Class to load each tile and neighboring tiles data, perform the registration and segmentation.
    A tileGraph object must be initialized and passed for computing the registration.

    Tile post processing is done on APR data, so if the input data is tiff it is first converted.
    """
    def __init__(self, tile):
        # TODO: here the data of each tile is passed as a dictionary. Maybe there is a less redundant or cleaner way
        # to achieve this.
        self.path = tile['path']
        self.row = tile['row']
        self.col = tile['col']
        self.type = tile['type']
        self.neighbors = tile['neighbors']
        self.neighbors_path = tile['neighbors_path']
        self.overlap = tile['overlap']
        self.frame_size = tile['frame_size']

        # Load tile data and neighbors data.
        self.data = self._load_tile(self.path)
        self.data_neighbors = self._load_neighbors()

        # If data is not APR then convert it
        if self.type != 'apr':
            self._convert_to_apr()

        # Initialize attributs for segmentation
        self.path_classifier = None
        self.f_names = None

    def compute_registration(self, tgraph):
        """
        Compute the pair-wise registration for a given tile with all its neighbors (EAST and SOUTH to avoid
        the redundancy).

        Parameters
        ----------
        tgraph: (tileGraph object) stores pair-wise registration to further perform the global optimization.

        """
        for v, coords in zip(self.data_neighbors, self.neighbors):
            if self.row == coords[0] and self.col < coords[1]:
                # EAST
                reg, rel = self._compute_east_registration(v)

            elif self.col == coords[1] and self.row < coords[0]:
                # SOUTH
                reg, rel = self._compute_south_registration(v)

            else:
                raise TypeError('Error: couldn''t determine registration to perform.')

            tgraph.cgraph_from.append(np.ravel_multi_index([self.row, self.col], dims=(tgraph.nrow, tgraph.ncol)))
            tgraph.cgraph_to.append(np.ravel_multi_index([coords[0], coords[1]], dims=(tgraph.nrow, tgraph.ncol)))
            # H=x, V=y, D=z
            tgraph.dH.append(reg[2])
            tgraph.dV.append(reg[1])
            tgraph.dD.append(reg[0])
            tgraph.relia_H.append(rel[2])
            tgraph.relia_V.append(rel[1])
            tgraph.relia_D.append(rel[0])

    def compute_segmentation(self, path_classifier=None):
        """
        Compute the segmentation (iLastik like) on the tile. Before attempting this, a model needs to be trained and
        saved.

        Parameters
        ----------
        path_classifier: (str) path to the trained model for the classification.

        """

        if path_classifier is None:
            raise ValueError('Error: no classifier path was given.')

        # Load classifier
        clf = load(path_classifier)

        # Compute features
        # TODO: is there a way to adapt the computed features to the ones used to train the classifier?
        f = self._compute_features()
        apr = self.data[0]

        # Predict particle type (cell, membrane or brackground) for each cell with the trained model
        parts_pred = self._predict_on_APR(clf, f)
        # Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
        parts_cells = (parts_pred == 0)
        # Remove small holes to get the misclassified nuclei
        parts_cells = pyapr.numerics.transform.remove_small_holes(apr, parts_cells, min_volume=500)
        # Opening to better separate touching cells
        pyapr.numerics.transform.opening(apr, parts_cells, radius=1, inplace=True)
        # Apply connected component
        cc = pyapr.ShortParticles()
        pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)
        # Remove small objects
        pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=200)
        # Save segmentation results
        pyapr.io.write(os.path.join(self.path, 'segmentation.apr'), apr, cc)

    def _load_tile(self, path):
        """
        Load the current tile.
        """
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
        elif self.type == 'apr':
            apr = pyapr.APR()
            parts = pyapr.ShortParticles()
            pyapr.io.read(*glob(os.path.join(path, '*0.apr')), apr, parts)
            u = (apr, parts)
        else:
            raise TypeError('Error: image type {} not supported.'.format(self.type))
        return u

    def _load_neighbors(self):
        """
        Load the current tile neighbors.
        """
        u = []
        for neighbor in self.neighbors_path:
            u.append(self._load_tile(neighbor))
        return u

    def _convert_to_apr(self):
        """
        Converts input tile from pixel data to APR.
        """
        # TODO: have an automatic way to set the parameters.

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

    def _compute_features(self):
        """
        Returns the features computed on APR data for using with the classifier to produce the segmentation.
        """
        apr = self.data[0]
        parts = self.data[1]

        # Compute gradient along a dimension (Sobel filter). dimension can be 0, 1 or 2
        grad_x, grad_y, grad_z = self._compute_gradients(apr, parts)

        # Compute gradient magnitude (central finite differences)
        grad = self._compute_gradmag(apr, parts)

        # Compute local standard deviation around each particle
        local_std = self._compute_std(apr, parts, size=5)

        # Compute lvl for each particle
        lvl = self._particle_levels(apr, normalize=True)

        # Compute difference of Gaussian
        dog = self._gaussian_blur(apr, parts, sigma=3, size=22) - self._gaussian_blur(apr, parts, sigma=1.5, size=11)

        # Aggregate filters in a feature array
        f = np.vstack((np.array(parts, copy=True),
                       lvl,
                       grad_x,
                       grad_y,
                       grad_z,
                       grad,
                       local_std,
                       dog
                       )).T
        if self.f_names is None:
            self.f_names = ['Intensity',
                            'lvl',
                            'grad_x',
                            'grad_y',
                            'grad_z',
                            'grad_mag',
                            'local_std',
                            'dog'
                            ]
        return f

    def _compute_gradients(self, apr, parts, sobel=True):
        """
        Returns the gradients [dz, dx, dy] of APR data.
        """
        par = apr.get_parameters()
        dx = pyapr.FloatParticles()
        dy = pyapr.FloatParticles()
        dz = pyapr.FloatParticles()

        pyapr.numerics.gradient(apr, parts, dz, dimension=2, delta=par.dz, sobel=sobel)
        pyapr.numerics.gradient(apr, parts, dx, dimension=1, delta=par.dx, sobel=sobel)
        pyapr.numerics.gradient(apr, parts, dy, dimension=0, delta=par.dy, sobel=sobel)
        return dz, dx, dy

    def _compute_laplacian(self, apr, parts, sobel=True):
        """
        Returns the Laplacian of APR data.
        """
        # TODO: merge this with compute gradient to avoid computing it twice.
        par = apr.get_parameters()
        dz, dx, dy = self._compute_gradients(apr, parts, sobel)
        dx2 = pyapr.FloatParticles()
        dy2 = pyapr.FloatParticles()
        dz2 = pyapr.FloatParticles()
        pyapr.numerics.gradient(apr, dz, dz2, dimension=2, delta=par.dz, sobel=sobel)
        pyapr.numerics.gradient(apr, dx, dx2, dimension=1, delta=par.dx, sobel=sobel)
        pyapr.numerics.gradient(apr, dy, dy2, dimension=0, delta=par.dy, sobel=sobel)
        return dz + dx + dy

    def _compute_gradmag(self, apr, parts, sobel=True):
        """
        Returns the gradient magnitude of APR data.
        """
        # TODO: idem laplacian, this can probably be optimized.
        par = apr.get_parameters()
        gradmag = pyapr.FloatParticles()
        pyapr.numerics.gradient_magnitude(apr, parts, gradmag, deltas=(par.dz, par.dx, par.dy), sobel=True)
        return gradmag

    def _gaussian_blur(self, apr, parts, sigma=1.5, size=11):
        """
        Returns a gaussian filtered APR data.
        """
        stencil = pyapr.numerics.get_gaussian_stencil(size, sigma, 3, True)
        output = pyapr.FloatParticles()
        pyapr.numerics.filter.convolve_pencil(apr, parts, output, stencil, use_stencil_downsample=True,
                                              normalize_stencil=True, use_reflective_boundary=True)
        return output

    def _particle_levels(self, apr, normalize=True):
        """
        Returns the particle level of APR data.
        """
        lvls = pyapr.ShortParticles(apr.total_number_particles())
        lvls.fill_with_levels(apr)
        if normalize:
            lvls *= (1 / apr.level_max())
        return lvls

    def _compute_std(self, apr, parts, size=5):
        """
        Returns the local std of APR data.
        """
        dims = apr.org_dims()
        box_size = [size if d >= size else 1 for d in dims]
        locstd = pyapr.FloatParticles()
        pyapr.numerics.local_std(apr, parts, locstd, size=box_size)
        return locstd

    def _predict_on_APR(self, clf, x):
        """
        Predict particle class with the trained classifier clf on the precomputed features f.
        """
        # Predict on numpy array
        y_pred = clf.predict(x)

        # Transform numpy array to ParticleData
        parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

        return parts_pred

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

    def _compute_east_registration(self, v):
        """
        Compute the registration between the current tile and its eastern neighbor.
        """
        apr_1, parts_1 = self.data
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

        return self._get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                     [proj_zy2, proj_zx2, proj_yx2])

    def _compute_south_registration(self, v):
        """
        Compute the registration between the current tile and its southern neighbor.
        """
        apr_1, parts_1 = self.data
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

        return self._get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                     [proj_zy2, proj_zx2, proj_yx2])


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
            ax[0, i].imshow(reg_rel_map[i], cmap='gray')
            ax[0, i].set_title('Rel reg. map ' + d)
            ax[1, i].imshow(reg_abs_map[i], cmap='gray')
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


class tileViewer():
    """
    Class to display the registration and segmentation using Napari.
    """
    def __init__(self, tiles, tgraph, segmentation=False):
        self.tiles = tiles
        self.tgraph = tgraph
        self.nrow = tiles.nrow
        self.ncol = tiles.ncol
        self.loaded_ind = []
        self.loaded_tiles = {}
        self.segmentation = segmentation
        self.loaded_segmentation = {}

    def display_tiles(self, coords, level_delta=0, **kwargs):
        """
        Display the tiles with coordinates given in coords (np array).
        """
        # Check that coords is (n, 2) or (2, n)
        if coords.size == 2:
            coords = np.array(coords).reshape(1, 2)
        elif coords.shape[1] != 2:
            coords = coords.T
            if coords.shape[1] != 2:
                raise ValueError('Error, at least one dimension of coords should be of size 2.')

        # Compute layers to be displayed by Napari
        layers = []
        for i in range(coords.shape[0]):
            row = coords[i, 0]
            col = coords[i, 1]

            # Load tile if not loaded, else use cached tile
            ind = np.ravel_multi_index((row, col), dims=(self.nrow, self.ncol))
            if self._is_tile_loaded(row, col):
                apr, parts = self.loaded_tiles[ind]
                if self.segmentation:
                    mask = self.loaded_segmentation[ind]
            else:
                apr, parts = self._load_tile(row, col)
                self.loaded_ind.append(ind)
                self.loaded_tiles[ind] = apr, parts
                if self.segmentation:
                    apr, mask = self._load_segmentation(row, col)
                    self.loaded_segmentation[ind] = mask

            position = self._get_tile_position(row, col)
            if level_delta != 0:
                position = [x/level_delta**2 for x in position]
            layers.append(apr_to_napari_Image(apr, parts,
                                               mode='constant',
                                               name='Tile [{}, {}]'.format(row, col),
                                               translate=position,
                                               opacity=0.7,
                                               level_delta=level_delta,
                                               **kwargs))
            if self.segmentation:
                layers.append(apr_to_napari_Labels(apr, mask,
                                                  mode='constant',
                                                  name='Segmentation [{}, {}]'.format(row, col),
                                                  translate=position,
                                                  level_delta=level_delta,
                                                  opacity=0.7))

        # Display layers
        display_layers(layers)

    def _load_segmentation(self, row, col):
        """
        Load the segmentation for tile at position [row, col].
        """
        df = tgraph.database
        path = df[(df['row'] == row) & (df['col'] == col)]['path'].values[0]
        apr = pyapr.APR()
        parts = pyapr.ShortParticles()
        pyapr.io.read(*glob(os.path.join(path, 'segmentation.apr')), apr, parts)
        u = (apr, parts)
        return u

    def _is_tile_loaded(self, row, col):
        """
        Returns True is tile is loaded, False otherwise.
        """
        ind = np.ravel_multi_index((row, col), dims=(self.nrow, self.ncol))
        return ind in self.loaded_ind

    def _load_tile(self, row, col):
        """
        Load the tile at position [row, col].
        """
        df = tgraph.database
        path = df[(df['row'] == row) & (df['col'] == col)]['path'].values[0]
        if self.tiles.type == 'tiff2D':
            files = glob(os.path.join(path, '*.tif'))
            im = imread(files[0])
            u = np.zeros((len(files), *im.shape))
            u[0] = im
            files.pop(0)
            for i, file in enumerate(files):
                u[i+1] = imread(file)
            return self._get_apr(u)
        elif self.tiles.type == 'tiff3D':
            u = imread(*glob(os.path.join(path, '*.tif')))
            return self._get_apr(u)
        elif self.tiles.type == 'apr':
            apr = pyapr.APR()
            parts = pyapr.ShortParticles()
            pyapr.io.read(*glob(os.path.join(path, '*0.apr')), apr, parts)
            u = (apr, parts)
            return u
        else:
            raise TypeError('Error: image type {} not supported.'.format(self.type))

    def _get_apr(self, u):
        # TODO: remove this by saving APR at previous steps?
        # Parameters are hardcoded for now
        par = pyapr.APRParameters()
        par.auto_parameters = False  # really heuristic and not working
        par.sigma_th = 26.0
        par.grad_th = 3.0
        par.Ip_th = 253.0
        par.rel_error = 0.2
        par.gradient_smoothing = 2

        # Convert data to APR
        return pyapr.converter.get_apr(image=u, params=par, verbose=False)

    def _get_tile_position(self, row, col):
        """
        Parse tile position in the database.
        """
        df = self.tgraph.database
        tile_df = df[(df['row'] == row) & (df['col'] == col)]
        px = tile_df['ABS_H'].values[0]
        py = tile_df['ABS_V'].values[0]
        pz = tile_df['ABS_D'].values[0]

        return [pz, py, px]


if __name__=='__main__':
    from time import time
    path = r'/mnt/Data/wholebrain/multitile'
    t = time()
    t_ini = time()
    tiles = tileParser(path)
    print('Elapsed time parse data: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    tgraph = tileGraph(tiles)
    print('Elapsed time init tgraph: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    for tile in tiles:
        loaded_tile = tileLoader(tile)
        loaded_tile.compute_registration(tgraph)
        # loaded_tile.compute_segmentation(path_classifier=
                                         # r'/media/sf_shared_folder_virtualbox/PV_interneurons/classifiers/random_forest_n100.joblib')
    print('Elapsed time load, segment, and compute pairwise reg: {:.2f} s.'.format(time() - t))

    t = time()
    tgraph.build_sparse_graphs()
    print('Elapsed time build sparse graph: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    tgraph.optimize_sparse_graphs()
    print('Elapsed time optimize graph: {:.2f} ms.'.format((time() - t)*1000))
    tgraph.plot_min_trees(annotate=True)
    t = time()
    reg_rel_map, reg_abs_map = tgraph.produce_registration_map()
    print('Elapsed time reg map: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    tgraph.build_database(tiles)
    print('Elapsed time build database: {:.2f} ms.'.format((time() - t)*1000))
    t = time()
    tgraph.save_database(os.path.join(path, 'registration_results.csv'))
    print('Elapsed time save database: {:.2f} ms.'.format((time() - t)*1000))

    print('\n\nTOTAL elapsed time: {:.2f} s.'.format(time() - t_ini))

    viewer = tileViewer(tiles, tgraph, segmentation=False)
    coords = []
    for i in range(2):
        for j in range(2):
            coords.append([i, j])
    coords = np.array(coords)
    viewer.display_tiles(coords, level_delta=-2, contrast_limits=[0, 15000])

    cr = []
    for i in range(4):
        cr.append(viewer.loaded_tiles[i][0].computational_ratio())
    print(np.mean(cr))