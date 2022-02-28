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
        self.acq_param = None
        self.n_channels = n_channels
        self.nrow, self.ncol, self.n_planes = self._parse_acquisition_settings()
        self.frame_size = 2048
        self.tile_processed = 0
        self.n_tiles = self.nrow * self.ncol
        self.type = 'clearscope'
        self.current_tile = 1
        self.current_channel = 0

        # Converter attributes
        self.converter = None
        self.lazy_loading = None
        self.compression = False
        self.bg = None
        self.quantization_factor = None
        self.folder_apr = None

        # Stitcher attributes
        self.stitcher = None
        self.overlap_h = None
        self.overlap_v = None
        self.n_vertex = None
        self.folder_max_projs = None

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

                tile.load_tile()

                # Convert tile
                if self.converter is not None:
                    self._convert_to_apr(tile)
                    self._check_conversion(tile)

                if self.stitcher is True:
                    self._pre_stitch(tile)

                self._update_next_tile()

                self.tile_processed += 1
            else:
                sleep(1)

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

    def activate_stitching(self,
                           overlap_h: (int, float),
                           overlap_v: (int, float),
                           ):

        self.overlap_h = overlap_h
        self.overlap_v = overlap_v
        self.stitcher = True

        # Safely create folder to save max projs
        self.folder_max_projs = os.path.join(self.path, 'max_projs')
        Path(self.folder_max_projs).mkdir(parents=True, exist_ok=True)

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

    def _parse_acquisition_settings(self):

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

        nrow = int(self.acq_param['ScanGridY'])
        ncol = int(self.acq_param['ScanGridY'])
        n_planes =int(self.acq_param['StackDepths'])

        print('\nAcquisition parameters:'
              '\nnumber of row: {}'
              '\nnumber of col: {}'
              '\nnumber of planes: {}'
              '\nnumber of channels: {}'.format(nrow, ncol, n_planes,
                                                                                               self.n_channels))

        return nrow, ncol, n_planes

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

        return pipapr.loader.tileLoader(path=path,
                                        row=row,
                                        col=col,
                                        ftype=self.type,
                                        neighbors=None,
                                        neighbors_tot=None,
                                        neighbors_path=None,
                                        frame_size=2048,
                                        folder_root=self.path,
                                        channel=channel)

    def _pre_stitch(self, tile):

       # Max project current tile on the overlaping area.

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