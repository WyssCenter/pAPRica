"""
Submodule containing classes and functions relative to **batch processing**.

This submodule introduce the notion of multichannel acquisition which can be either the folder structure resulting
from a given microscope acquisition or a converted acquisition (usually living in the original acquisition folder).

The default (converted) folder is saved in the first acquisition folder with the following structure:

        APR/ch0/0_0.apr
                0_1.apr
                ...
                n_m.apr
            ch1/0_0.apr
                0_1.apr
                ...
                n_m.apr
            ...
            chx/0_0.apr
                0_1.apr
                ...
                n_m.apr

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import os
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from glob import glob
from tqdm import tqdm
import pyapr

import pipapr


class multiChannelAcquisition():

    def __init__(self, path):
        """
        Class to store a multichannel acquisition. If data was converted to APR, then this class is instantiated with
        the APR rather than the raw data.

        Parameters
        ----------
        path: str
            path to folder containing the acquisition.
        """
        self.path = path
        self.n_channels = pipapr.parser.get_number_of_channels(self.path)
        # Check if apr is available
        self.is_apr_available = os.path.exists(os.path.join(path, 'APR'))
        if self.is_apr_available:
            self.acq_type = self._get_acq_type()
            self.tiles_list = self._get_tiles_list(verbose=False)
            self.overlap_v, self.overlap_h = self.tiles_list[0].get_overlap()
            self.acq_type = 'apr'
            self.path = os.path.join(self.path, 'APR')
            self.tiles_list = self._get_tiles_list()
        else:
            self.acq_type = self._get_acq_type()
            self.tiles_list = self._get_tiles_list()
            self.overlap_v, self.overlap_h = self.tiles_list[0].get_overlap()




    def convert_all_channels(self,
                             Ip_method='black_corner',
                             Ip=None,
                             force_convert=False,
                             rel_error=0.2,
                             gradient_smoothing=2,
                             dx=1,
                             dy=1,
                             dz=1,
                             lazy_loading=True,
                             tree_mode='mean',):
        """
        Function to convert all the channel to APR. The intensity threshold `Ip_th` is automatically determined
        using the provided method or using the given

        Parameters
        ----------
        Ip_method: str
            Method to compute Ip_th automatically
        Ip: list
            List of Ip_th to be used for the conversion
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
        tree_mode: str ('mean' or 'max')
            controls how downsampled particles are computed. Either the mean or the max is taken.

        Returns
        -------
        None
        """

        for i, tiles in enumerate(self.tiles_list):
            # Safely create folder to save apr data
            folder_apr = os.path.join(self.path, 'APR', 'ch{}'.format(tiles.channel))
            Path(folder_apr).mkdir(parents=True, exist_ok=True)

            for tile in tiles:
                if force_convert or not os.path.exists(os.path.join(folder_apr,
                                                                    '{}_{}.apr'.format(tile.row, tile.col))):
                        # Either fetch Ip_th or automatically compute it
                        if Ip is not None:
                            Ip_th = Ip[i]
                        else:
                            Ip_th = self._get_Ip_th(tiles, method=Ip_method)

                        tile.load_tile()

                        # Set parameters
                        par = pyapr.APRParameters()
                        par.Ip_th = Ip_th
                        par.rel_error = rel_error
                        par.dx = dx
                        par.dy = dy
                        par.dz = dz
                        par.gradient_smoothing = gradient_smoothing
                        par.auto_parameters = True

                        # Convert tile to APR and save
                        apr = pyapr.APR()
                        parts = pyapr.ShortParticles()
                        converter = pyapr.converter.FloatConverter()
                        converter.set_parameters(par)
                        converter.verbose = True
                        converter.get_apr(apr, tile.data)
                        parts.sample_image(apr, tile.data)

                        if lazy_loading:
                            if tree_mode == 'mean':
                                tree_parts = pyapr.tree.fill_tree_mean(apr, parts)
                            elif tree_mode == 'max':
                                tree_parts = pyapr.tree.fill_tree_max(apr, parts)
                        else:
                            tree_parts = None

                        # Save converted data
                        filename = '{}_{}.apr'.format(tile.row, tile.col)
                        pyapr.io.write(os.path.join(folder_apr, filename), apr, parts, tree_parts=tree_parts)
                else:
                    print('Tile {}_{}.apr already exists, it will not be converted. If you want to force the '
                          'conversion please use ''force_convert'' flag'.format(tile.row, tile.col))

            self.acq_type = 'apr'
            self.path = os.path.join(self.path, 'APR')
            self.tiles_list = self._get_tiles_list()

    def stitch_acq(self,
                   channel):

        tiles = self.tiles_list[channel]
        stitcher = pipapr.tileStitcher(tiles, overlap_h=self.overlap_h, overlap_v=self.overlap_v)

        tile = tiles[0]
        tile.lazy_load_tile(level_delta=0)
        z = int(tile.lazy_data.shape[0] / 2)

        stitcher.set_z_range(z_begin=z-50, z_end=z+50)
        stitcher.set_overlap_margin(margin=10)

        stitcher.compute_registration()
        stitcher.save_database(os.path.join(self.path, 'registration_results.csv'))

        self.database = stitcher.database
        self.stitcher = stitcher


    def reconstruct_3D_all_channels(self,
                                    downsample=16,
                                    ):

        for tiles in self.tiles_list:
            merger = pipapr.stitcher.tileMerger(tiles, self.database)
            merger.set_downsample(downsample)
            merger.merge_max()
            imsave(os.path.join(tiles.path, '3D_reconstruction.tif'), merger.merged_data)

    # def stitch_single_channel(self,
    #                           channel=0):
    #
    #     # Check if apr is available
    #     self.is_apr_available = os.path.exists(os.path.join(self.path, 'APR'))
    #     # If data was previously converted to APR then the microscopy acquisition is the APR folder containing the
    #     # folder for each channel.
    #     if self.is_apr_available:
    #         self.acq_type = 'apr'
    #         self.path = os.path.join(path, 'APR')
    #     else:
    #
    #     from skimage.io import imsave
    #
    #     tiles = self.tiles_list[channel]
    #     overlap_v, overlap_h = tiles.get_overlap()
    #     stitcher = pipapr.tileStitcher(tiles, overlap_h=overlap_h, overlap_v=overlap_v)
    #
    #     tile = tiles[0]
    #     tile.lazy_load_tile(level_delta=0)
    #     z = int(tile.lazy_data.shape[0] / 2)
    #
    #     stitcher.set_z_range(z_begin=z-50, z_end=z+50)
    #     stitcher.set_overlap_margin(margin=10)
    #
    #     stitcher.compute_registration()
    #     stitcher.save_database()
    #
    #     u = stitcher.reconstruct_slice(plot=False)
    #     imsave(os.path.join(tiles.path, 'middle_plane_stitched.tif'), u, check_contrast=False)

    def _get_tiles_list(self, **kwargs):
        """
        Function to get the list of tiles (one tileParser object for each channel)

        """
        if self.acq_type == 'apr':
            return [pipapr.tileParser(f, ftype='apr', **kwargs) for f in sorted(glob(os.path.join(self.path, 'ch*/')))]
        else:
            return [pipapr.autoParser(self.path, channel=x, **kwargs) for x in range(pipapr.parser.get_number_of_channels(self.path))]

    def _get_acq_type(self):
        """
        Get the acquisition type (type of microscope used to acquire the data).

        Returns
        -------
        tiles.type: str
            microscope used to acquire the data
        """

        tiles = pipapr.autoParser(self.path, verbose=False)
        return tiles.type

    def _get_Ip_th(self, tiles, method):
        """
        Function to compute the intensity threshold value (Ip_th) automatically.

        Parameters
        ----------
        tiles: pipapr.parser.tileParser
            tileParser object to compute Ip_th for.
        method: str
            method to compute Ip_th automatically:
            - 'black_corner': compute Ip_th as the mean of the first tile in the top left corner (50 x 50 pixels). This
                    methods works if there is no signal in the top left part of the acquisition (which is usually the
                    case).

        Returns
        -------

        """
        if method == 'black_corner':
            u = imread(sorted(glob(tiles.path_list[0] + '*.tif'))[0])
            return int(np.mean(u[:50, :50]))
        else:
            raise ValueError('Error: unknown method for computing Ip_th')

    def __getitem__(self, item):
        return self.tiles_list[item]
