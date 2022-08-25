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

import numpy as np
import pandas as pd
from skimage.filters import gaussian
from skimage.io import imread, imsave
from tqdm import tqdm
import re
from glob import glob
from warnings import warn

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
        self.is_apr_available = os.path.exists(os.path.join(path, 'APR'))
        # If data was previously converted to APR then the microscopy acquisition is the APR folder containing the
        # folder for each channel.
        if self.is_apr_available:
            self.acq_type = 'apr'
            self.path = os.path.join(path, 'APR')
        else:
            self.acq_type = self._get_acq_type()

        self.tiles_list = self._get_tiles_list()

    def convert_all_channels(self, Ip_method='black_edge', Ip=None, **kwargs):
        """
        Function to convert all the channel to APR. The intensity threshold `Ip_th` is automatically determined
        using the provided method or using the given

        Parameters
        ----------
        Ip_method: str
            Method to compute Ip_th automatically
        Ip: list
            List of Ip_th to be used for the conversion
        kwargs:
            kwargs for `batch_convert_to_apr()` method

        Returns
        -------
        None
        """
        for i, tiles in enumerate(self.tiles_list):
            if tiles.type != 'apr':
                if Ip_th is None:
                    Ip_th = Ip[i]
                else:
                    Ip_th = self._get_Ip_th(tiles, method=Ip_method)
                converter = pipapr.tileConverter(tiles)
                converter.batch_convert_to_apr(Ip_th=Ip_th, **kwargs)

    def _get_tiles_list(self):
        """
        Function to get the list of tiles (one tileParser object for each channel)

        """
        if self.acq_type == 'apr':
            return [pipapr.tileParser(f, ftype='apr') for f in sorted(glob(os.path.join(self.path, 'ch*/')))]
        else:
            return [pipapr.autoParser(self.path, channel=x) for x in range(pipapr.parser.get_channel_number(self.path))]

    def _get_acq_type(self):

        tiles = pipapr.autoParser(self.path, verbose=False)
        return tiles.type

    def _get_Ip_th(self, tiles, method):

        if method == 'black_edge':
            u = imread(sorted(glob(tiles.path_list[0] + '*.tif'))[0])
            return int(np.mean(u[:50, :50]))
        else:
            raise ValueError('Error: unknown method for computing Ip_th')

    def __getitem__(self, item):
        return self.tiles_list[item]
