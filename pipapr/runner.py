"""
Submodule containing classes and functions relative to the running pipeline.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import re
from glob import glob
import os
import numpy as np
import pipapr
from pipapr.stitcher import _get_max_proj_apr, _get_proj_shifts
from time import sleep
import pyapr
from skimage.io import imread
from tqdm import tqdm
from pathlib import Path
import pandas as pd

class clearscopeRunningPipeline():

    def __init__(self, path, n_channels):
        """
        Constructor for the clearscopeRunningPipeline.

        Parameters
        ----------
        path: str
            Path of the acquisition. It has to be the acquisition folder (root of /0001/ folder).
        n_channels: int
            Number of channels that will be acquired. This will be parsed/guessed in the future.
        """

        # runningPipeline attributes
        self.path = os.path.join(path, '0001')
        self.folder_settings, self.name_acq = os.path.split(path)
        self.frame_size = 2048
        self.n_channels = n_channels
        self.tile_processed = 0
        self.type = 'clearscope'
        self.current_tile = 1
        self.current_channel = 0

        # Parsing acquisition parameters
        self.acq_param = None
        self.nrow = None
        self.ncol = None
        self.n_planes = None
        self.overlap_v = None
        self.overlap_h = None
        self._parse_acquisition_settings()
        self.n_tiles = self.nrow * self.ncol
        self.projs = np.empty((self.nrow, self.ncol), dtype=object)

        # Converter attributes
        self.converter = None
        self.lazy_loading = None
        self.compression = False
        self.bg = None
        self.quantization_factor = None
        self.folder_apr = None

        # Stitcher attributes
        self.stitcher = None
        self.n_vertex = None
        self.folder_max_projs = None

        self.mask = False
        self.threshold = None

        self.segment = False
        self.segmenter = None

        self.reg_x = int(self.frame_size*0.05)
        self.reg_y = int(self.frame_size*0.05)
        self.reg_z = 20

        self.z_begin = None
        self.z_end = None

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

    def run(self):
        """
        Start the running pipeline. It is basically a loop waiting for each tile to be saved at the specified path.

        Returns
        -------
        None
        """

        while self.tile_processed < self.n_tiles:

            is_available, tile = self._is_new_tile_available()

            if is_available:

                print('\nNew tile available: {}\nrow: {}\ncol {}\nchannel {}'.format(tile.path,
                                                                                      tile.row,
                                                                                      tile.col,
                                                                                      tile.channel))

                # We check if the APR file is already available (e.g. something crashed and we restart the pipeline)
                apr, parts = self._check_for_apr_file(tile)
                if apr is None:
                    tile.load_tile()

                    # Convert tile
                    if self.converter is not None:
                        self._convert_to_apr(tile)
                        self._check_conversion(tile)
                else:
                    tile.apr = apr
                    tile.parts = parts

                if self.stitcher is True:
                    self._pre_stitch(tile)

                self._update_next_tile()

                self.tile_processed += 1
            else:
                sleep(1)

        if self.stitcher:
            self._build_sparse_graphs()
            self._optimize_sparse_graphs()
            _, _ = self._produce_registration_map()
            self._build_database()
            self._print_info()

    def activate_conversion(self,
                             Ip_th=108,
                             rel_error=0.2,
                             gradient_smoothing=2,
                             dx=1,
                             dy=1,
                             dz=1,
                             lazy_loading=True):
        """
        Activate conversion for the running pipeline.

        Parameters
        ----------
        Ip_th: int
            Intensity threshold
        rel_error: float in [0, 1[
            relative error bound
        gradient_smoothing: (float)
            B-Spline smoothing parameter (typically between 0 (no smoothing) and 10 (LOTS of smoothing)
        dx: float
            PSF size in x, used to compute the gradient
        dy: float
            PSF size in y, used to compute the gradient
        dz: float
            PSF size in z, used to compute the gradient
        lazy_loading: bool
            if lazy_loading is true then the converter save mean tree particle which are necessary for lazy loading of
            the APR. It will require about 1/7 more storage.

        Returns
        -------
        None
        """

        # Store parameters
        self.lazy_loading = lazy_loading

        # Safely create folder to save apr data
        for i in range(self.n_channels):
            self.folder_apr = os.path.join(self.path, 'APR')
            Path(os.path.join(self.folder_apr, 'ch{}'.format(i))).mkdir(parents=True, exist_ok=True)

        # Set parameters
        par = pyapr.APRParameters()
        par.Ip_th = Ip_th
        par.rel_error = rel_error
        par.dx = dx
        par.dy = dy
        par.dz = dz
        par.gradient_smoothing = gradient_smoothing
        par.auto_parameters = True

        # Create converter object
        self.converter = pyapr.converter.FloatConverter()
        self.converter.set_parameters(par)
        self.converter.verbose = True

    def set_compression(self, quantization_factor=1, bg=108):
        """
        Activate B3D compression for saving tiles.

        Parameters
        ----------
        quantization_factor: int
            quantization factor: the higher, the more compressed (refer to B3D paper for more detail).
        bg: int
            background value: any value below this threshold will be set to the background value. This helps
            save up space by having the same value for the background (refer to B3D paper for more details).

        Returns
        -------
        None
        """

        self.compression = True
        self.bg = bg
        self.quantization_factor = quantization_factor

    def deactivate_compression(self):
        """
        Deactivate B3D compression when saving particles.

        Returns
        -------
        None
        """

        self.compression = False
        self.bg = None
        self.quantization_factor = None

    def activate_stitching(self, channel):

        self.stitcher = True
        self.stitched_channel = channel

        # Safely create folder to save max projs
        self.folder_max_projs = os.path.join(self.path, 'max_projs')
        Path(os.path.join(self.folder_max_projs, 'ch{}'.format(channel))).mkdir(parents=True, exist_ok=True)

    def _check_for_apr_file(self, tile):

        apr_path = os.path.join(self.folder_apr, 'ch{}'.format(tile.channel),
                                                    '{}_{}.apr'.format(tile.row, tile.col))
        if os.path.exists(apr_path):
            apr, parts = pyapr.io.read(apr_path)
            print('Tile {}_{}.apr already exists!'.format(tile.row, tile.col))
            return apr, parts
        else:
            return None, None

    def _parse_acquisition_settings(self):
        """
        Function that parses the setting txt file created by cleascope software at the begining of the acquisition
        and automatically extract the required parameters for the running pipeline to work.

        Returns
        -------
        None
        """

        print('Waiting for AcquireSettings.txt file in {}'.
              format(os.path.join(self.folder_settings, '{}_AcquireSettings.txt'.format(self.name_acq))))

        files = glob(os.path.join(self.folder_settings, '{}_AcquireSettings.txt'.format(self.name_acq)))
        while files == []:
            sleep(1)
            files = glob(os.path.join(self.folder_settings, '{}_AcquireSettings.txt'.format(self.name_acq)))

        path = files[0]
        print('File found: {}'.format(path))


        with open(path) as f:
            lines = f.readlines()

        self.acq_param = {}
        for l in lines:
            pattern_matched = re.match('^(\w*) = (.*)$', l)
            if pattern_matched is not None:
                if pattern_matched.group(2).isnumeric():
                    self.acq_param[pattern_matched.group(1)] = float(pattern_matched.group(2))
                elif pattern_matched.group(2) == 'True':
                    self.acq_param[pattern_matched.group(1)] = True
                elif pattern_matched.group(2) == 'False':
                    self.acq_param[pattern_matched.group(1)] = False
                else:
                    self.acq_param[pattern_matched.group(1)] = pattern_matched.group(2)

        self.nrow = int(self.acq_param['ScanGridY'])
        self.ncol = int(self.acq_param['ScanGridY'])
        self.n_planes =int(self.acq_param['StackDepths'])

        self.expected_overlap_v = self.acq_param['VSThrowAwayYBottom']
        self.expected_overlap_h = self.acq_param['VSThrowAwayXRight']

        self.overlap_h = int(self.expected_overlap_h*1.2)
        if self.expected_overlap_h > self.frame_size:
            self.expected_overlap_h = self.frame_size
        self.overlap_v = int(self.expected_overlap_v*1.2)
        if self.expected_overlap_v > self.frame_size:
            self.expected_overlap_v = self.frame_size

        print('\nAcquisition parameters:'
              '\n- number of row: {}'
              '\n- number of col: {}'
              '\n- number of planes: {}'
              '\n- number of channels: {}'
              '\n- horizontal overlap: {:0.2f}%'
              '\n- vertical overlap: {:0.2f}%'
              .format(self.nrow, self.ncol, self.n_planes, self.n_channels,
                      self.expected_overlap_h/self.frame_size*100,
                      self.expected_overlap_v/self.frame_size*100))

    def _is_new_tile_available(self):
        """
        Checks if a new tile is available for processing.

        Returns
        -------
        is_available: bool
            True if a new tile is available, False otherwise
        tile: tileLoader
            tile object is available, None otherwise
        """

        expected_tile = os.path.join(self.path, '000000_{:06d}___{}c/'.format(self.current_tile, self.current_channel))
        path = glob(expected_tile)

        if path == []:
            return False, None
        elif len(path) == 1:
            # Store current tile coordinate
            files = glob(os.path.join(expected_tile, '*.tif'))
            if len(files) < self.n_planes:
                return False, None
            else:
                tile = self._get_tile(expected_tile)

            return True, tile
        else:
            raise TypeError('Error: multiple tiles were found.')

    def _get_row_col(self, path):
        """
        Get ClearScope tile row and col position given the tile number.

        Parameters
        ----------
        n: int
            ClearScope tile number

        Returns
        -------
        row: int
            row number
        col: int
            col number
        """

        pattern_search = re.findall('\d{6}_(\d{6})___\dc', path)

        if pattern_search != []:
            n = int(pattern_search[0])

        col = np.absolute(np.mod(n - self.ncol - 1, 2 * self.ncol) - self.ncol + 0.5) + 0.5
        row = np.ceil(n / self.ncol)

        col = int(col-1)
        row = int(row-1)

        return row, col

    def _get_channel(self, path):

        pattern_search = re.findall('\d{6}_\d{6}___(\d)c', path)

        if pattern_search != []:
            return int(pattern_search[0])

    def _update_next_tile(self):
        """
        Update next tile coordinates given the expected pattern.

        Returns
        -------
        None
        """
        if self.current_channel == self.n_channels-1:
            self.current_tile += 1
            self.current_channel = 0
        else:
            self.current_channel += 1

    def _get_tile(self, path):
        """
        Returns the tile at the given path.

        Parameters
        ----------
        path: string
            tile path

        Returns
        -------
        tile: tileLoader
            tile object
        """

        row, col = self._get_row_col(path)
        channel = self._get_channel(path)

        # If row is even then neighbors are west and north
        if row % 2 == 0:
            # If first row then it is only west
            if row == 0:
                if col > 0:
                    neighbors = [[row, col - 1]]
                else:
                    neighbors = None
            # Else it is also north
            else:
                if col > 0:
                    neighbors = [[row, col - 1], [row - 1, col]]
                # Except for first column it is only north
                else:
                    neighbors = [[row - 1, col]]
        # If row is odd then neighbors are north and east
        else:
            if col < self.ncol-1:
                neighbors = [[row - 1 , col], [row, col + 1]]
            # If last column then there is no east neighbor
            else:
                neighbors = [[row - 1 , col]]

        return pipapr.loader.tileLoader(path=path,
                                        row=row,
                                        col=col,
                                        ftype=self.type,
                                        neighbors=neighbors,
                                        neighbors_tot=None,
                                        neighbors_path=None,
                                        frame_size=2048,
                                        folder_root=self.path,
                                        channel=channel)

    def _pre_stitch(self, tile):

        if tile.channel == self.stitched_channel:
            # Max project current tile on the overlaping area.
            self._project_tile(tile)
            # Compute pair-wise registration with existing neighbors
            if tile.neighbors is not None:
                self._register_tile(tile)

    def _register_tile(self, tile):
        proj1 = self.projs[tile.row, tile.col]

        for coords in tile.neighbors:
            proj2 = self.projs[coords[0], coords[1]]

            if tile.row == coords[0]:
                if tile.col < coords[1]:
                    # EAST
                    if self.mask:
                        reg, rel = _get_masked_proj_shifts(proj1['east'], proj2['west'], threshold=self.threshold)
                    else:
                        reg, rel = _get_proj_shifts(proj1['east'], proj2['west'])
                else:
                    # WEST
                    if self.mask:
                        reg, rel = _get_masked_proj_shifts(proj1['west'], proj2['east'], threshold=self.threshold)
                    else:
                        reg, rel = _get_proj_shifts(proj1['west'], proj2['east'])

            elif tile.col == coords[1]:
                if tile.row < coords[0]:
                    # SOUTH
                    if self.mask:
                        reg, rel = _get_masked_proj_shifts(proj1['south'], proj2['north'], threshold=self.threshold)
                    else:
                        reg, rel = _get_proj_shifts(proj1['south'], proj2['north'])
                else:
                    # NORTH
                    if self.mask:
                        reg, rel = _get_masked_proj_shifts(proj1['north'], proj2['south'], threshold=self.threshold)
                    else:
                        reg, rel = _get_proj_shifts(proj1['north'], proj2['south'])

            else:
                raise TypeError('Error: couldn''t determine registration to perform.')

            self.cgraph_from.append(np.ravel_multi_index([tile.row, tile.col],
                                                         dims=(self.nrow, self.ncol)))
            self.cgraph_to.append(np.ravel_multi_index([coords[0], coords[1]],
                                                       dims=(self.nrow, self.ncol)))

            # Regularize in case of aberrant displacements
            reg, rel = self._regularize(reg, rel)

            # H=x, V=y, D=z
            self.dH.append(reg[2])
            self.dV.append(reg[1])
            self.dD.append(reg[0])
            self.relia_H.append(rel[2])
            self.relia_V.append(rel[1])
            self.relia_D.append(rel[0])

    def _project_tile(self, tile):

        proj = {}
        if tile.col + 1 < self.ncol:
            # EAST 1
            patch = pyapr.ReconPatch()
            patch.y_begin = self.frame_size - self.overlap_h
            if self.z_begin is None:
                proj['east'] = _get_max_proj_apr(tile.apr, tile.parts, patch, plot=False)
            else:
                patch_yx = pyapr.ReconPatch()
                patch_yx.y_begin = self.frame_size - self.overlap_h
                patch_yx.z_begin = self.z_begin
                patch_yx.z_end = self.z_end
                proj['east'] = _get_max_proj_apr(tile.apr, tile.parts, patch=patch, patch_yx=patch_yx,
                                                      plot=False)
            for i, d in enumerate(['zy', 'zx', 'yx']):
                np.save(os.path.join(self.folder_max_projs,
                                     'ch{}'.format(tile.channel),
                                    '{}_{}_east_{}.npy'.format(tile.row, tile.col, d)), proj['east'][i])
        if tile.col - 1 >= 0:
            # EAST 2
            patch = pyapr.ReconPatch()
            patch.y_end = self.overlap_h
            if self.z_begin is None:
                proj['west'] = _get_max_proj_apr(tile.apr, tile.parts, patch, plot=False)
            else:
                patch_yx = pyapr.ReconPatch()
                patch_yx.y_end = self.overlap_h
                patch_yx.z_begin = self.z_begin
                patch_yx.z_end = self.z_end
                proj['west'] = _get_max_proj_apr(tile.apr, tile.parts, patch=patch, patch_yx=patch_yx,
                                                      plot=False)
            for i, d in enumerate(['zy', 'zx', 'yx']):
                np.save(os.path.join(self.folder_max_projs,
                                     'ch{}'.format(tile.channel),
                                    '{}_{}_west_{}.npy'.format(tile.row, tile.col, d)), proj['west'][i])
        if tile.row + 1 < self.nrow:
            # SOUTH 1
            patch = pyapr.ReconPatch()
            patch.x_begin = self.frame_size - self.overlap_v
            if self.z_begin is None:
                proj['south'] = _get_max_proj_apr(tile.apr, tile.parts, patch, plot=False)
            else:
                patch_yx = pyapr.ReconPatch()
                patch_yx.x_begin = self.frame_size - self.overlap_v
                patch_yx.z_begin = self.z_begin
                patch_yx.z_end = self.z_end
                proj['south'] = _get_max_proj_apr(tile.apr, tile.parts, patch=patch, patch_yx=patch_yx,
                                                       plot=False)
            for i, d in enumerate(['zy', 'zx', 'yx']):
                np.save(os.path.join(self.folder_max_projs,
                                     'ch{}'.format(tile.channel),
                                    '{}_{}_south_{}.npy'.format(tile.row, tile.col, d)), proj['south'][i])
        if tile.row - 1 >= 0:
            # SOUTH 2
            patch = pyapr.ReconPatch()
            patch.x_end = self.overlap_v
            if self.z_begin is None:
                proj['north'] = _get_max_proj_apr(tile.apr, tile.parts, patch, plot=False)
            else:
                patch_yx = pyapr.ReconPatch()
                patch_yx.x_end = self.overlap_v
                patch_yx.z_begin = self.z_begin
                patch_yx.z_end = self.z_end
                proj['north'] = _get_max_proj_apr(tile.apr, tile.parts, patch=patch, patch_yx=patch_yx,
                                                       plot=False)
            for i, d in enumerate(['zy', 'zx', 'yx']):
                np.save(os.path.join(self.folder_max_projs,
                                     'ch{}'.format(tile.channel),
                                    '{}_{}_north_{}.npy'.format(tile.row, tile.col, d)), proj['north'][i])

        self.projs[tile.row, tile.col] = proj

    def _check_conversion(self, tile):
        """
        Checks that conversion is ok, if not it should keep the original data.

        Parameters
        ----------
        tile: tileLoader
            tile object to try if conversion worked as expected using facy metrics.

        Returns
        -------
        None
        """

        conversion_ok = False

        # TODO: implement a way to check if conversion is ok.

        if conversion_ok:
            tile._erase_from_disk()

    def _convert_to_apr(self, tile):
        """
        Convert the given tile to APR.

        Parameters
        ----------
        tile: tileLoader
            tile object to be converted to APR.

        Returns
        -------
        None
        """

        apr = pyapr.APR()
        parts = pyapr.ShortParticles()
        self.converter.get_apr(apr, tile.data)
        parts.sample_image(apr, tile.data)

        if self.compression:
            parts.set_compression_type(1)
            parts.set_quantization_factor(self.quantization_factor)
            parts.set_background(self.bg)

        if self.lazy_loading:
            tree_parts = pyapr.ShortParticles()
            pyapr.numerics.fill_tree_mean(apr, parts, tree_parts)
        else:
            tree_parts = None

        # Save converted data
        filename = '{}_{}.apr'.format(tile.row, tile.col)
        pyapr.io.write(os.path.join(self.folder_apr, 'ch{}'.format(tile.channel), filename),
                       apr, parts, tree_parts=tree_parts)

        tile.apr = apr
        tile.parts = parts

    def set_regularization(self, reg_x, reg_y, reg_z):
        """
        Set the regularization for the stitching to prevent aberrant displacements.

        Parameters
        ----------
        reg_x: int
            if the horizontal displacement computed in the pairwise registration for any tile is greater than
            reg_x (in pixel unit) then the expected displacement (from motor position) is taken.
        reg_y: int
            if the horizontal displacement computed in the pairwise registration for any tile is greater than
            reg_z (in pixel unit) then the expected displacement (from motor position) is taken.
        reg_z: int
            if the horizontal displacement computed in the pairwise registration for any tile is greater than
            reg_z (in pixel unit) then the expected displacement (from motor position) is taken.

        Returns
        -------
        None
        """

        self.reg_x = reg_x
        self.reg_y = reg_y
        self.reg_z = reg_z

    def _regularize(self, reg, rel):
        """
        Remove too large displacement and replace them with expected one with a large uncertainty.

        """
        if np.abs(reg[2] - (self.overlap_h - self.expected_overlap_h)) > self.reg_x:
            reg[2] = (self.overlap_h - self.expected_overlap_h)
            rel[2] = 2
        if np.abs(reg[1] - (self.overlap_v - self.expected_overlap_v)) > self.reg_y:
            reg[1] = (self.overlap_v - self.expected_overlap_v)
            rel[1] = 2
        if np.abs(reg[0]) > self.reg_z:
            reg[0] = 0
            rel[0] = 2

        return reg, rel

    def _print_info(self):
        """
        Display stitching result information.

        """
        overlap = np.median(np.diff(np.median(self.registration_map_abs[0], axis=0)))
        self.effective_overlap_h = (self.frame_size-overlap)/self.frame_size*100
        print('Effective horizontal overlap: {:0.2f}%'.format(self.effective_overlap_h))
        overlap = np.median(np.diff(np.median(self.registration_map_abs[1], axis=1)))
        self.effective_overlap_v = (self.frame_size-overlap)/self.frame_size*100
        print('Effective vertical overlap: {:0.2f}%'.format(self.effective_overlap_v))

        if np.abs(self.effective_overlap_v*self.frame_size/100-self.expected_overlap_v)>0.2*self.expected_overlap_v:
            warnings.warn('Expected vertical overlap is very different from the computed one, the registration '
                          'might be wrong.')
        if np.abs(self.effective_overlap_h*self.frame_size/100-self.expected_overlap_h)>0.2*self.expected_overlap_h:
            warnings.warn('Expected horizontal overlap is very different from the computed one, the registration '
                          'might be wrong.')

    def _build_sparse_graphs(self):
        """
        Build the sparse graph from the reliability and (row, col). This method needs to be called after the
        pair-wise registration has been performed for all neighbors pair.

        Returns
        -------
        None
        """

        csr_matrix_size = self.ncol*self.nrow
        self.graph_relia_H = csr_matrix((self.relia_H, (self.cgraph_from, self.cgraph_to)),
                                        shape=(csr_matrix_size, csr_matrix_size))
        self.graph_relia_V = csr_matrix((self.relia_V, (self.cgraph_from, self.cgraph_to)),
                                        shape=(csr_matrix_size, csr_matrix_size))
        self.graph_relia_D = csr_matrix((self.relia_D, (self.cgraph_from, self.cgraph_to)),
                                        shape=(csr_matrix_size, csr_matrix_size))

    def _optimize_sparse_graphs(self):
        """
        Optimize the sparse graph by computing the minimum spanning tree for each direction (H, D, V). This
        method needs to be called after the sparse graphs have been built.

        Returns
        -------
        None
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

        Returns
        -------
        None
        """

        if self.min_tree_H is None:
            raise TypeError('Error: minimum spanning tree not computed yet, please use optimize_sparse_graph()'
                            'before trying to compute the registration map.')

        # Relative registration
        # Initialize relative registration map
        reg_rel_map = np.zeros((3, self.nrow, self.ncol)) # H, V, D

        for i, min_tree in enumerate(['min_tree_H', 'min_tree_V', 'min_tree_D']):
            # Fill it by following the tree and getting the corresponding registration parameters
            node_array = depth_first_order(getattr(self, min_tree), i_start=self.cgraph_from[0],
                                           directed=False, return_predecessors=False)

            node_visited = [node_array[0]]

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
                # Get the corresponding relia and print a warning if it was regularized:
                relia = getattr(self, 'relia_' + min_tree[-1])[ind_graph]
                if relia == 2:
                    print('Aberrant pair-wise registration remaining after global optimization between tile ({},{}) '
                          'and tile ({},{})'.format(*np.unravel_index(node_from, shape=(self.nrow, self.ncol)),
                                                    *np.unravel_index(node_to, shape=(self.nrow, self.ncol))))
                # Update the local reg parameter in the 2D matrix
                if node_to > node_from[0]:
                    reg_rel_map[i, ind1, ind2] = d_neighbor + d
                else:
                    reg_rel_map[i, ind1, ind2] = d_neighbor - d
        self.registration_map_rel = reg_rel_map

        reg_abs_map = np.zeros_like(reg_rel_map)
        # H
        for x in range(reg_abs_map.shape[2]):
            reg_abs_map[0, :, x] = reg_rel_map[0, :, x] + x * (self.frame_size-self.overlap_h)
        # V
        for x in range(reg_abs_map.shape[1]):
            reg_abs_map[1, x, :] = reg_rel_map[1, x, :] + x * (self.frame_size-self.overlap_v)
        # D
        reg_abs_map[2] = reg_rel_map[2]
        self.registration_map_abs = reg_abs_map

        return reg_rel_map, reg_abs_map

    def _build_database(self):
        """
        Build the database for storing the registration parameters. This method needs to be called after
        the registration map has been produced.

        Returns
        -------
        None
        """

        if self.registration_map_rel is None:
            raise TypeError('Error: database can''t be build if the registration map has not been computed.'
                            ' Please use produce_registration_map() method first.')

        database_dict = {}
        for i in range(self.n_vertex):
            row = self.tiles[i].row
            col = self.tiles[i].col
            database_dict[i] = {'path': self.tiles[i].path,
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