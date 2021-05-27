"""
Module containing classes and functions relative to data loading.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from glob import glob
import os
from skimage.io import imread
import numpy as np
import pyapr
import re
from alive_progress import alive_bar


class tileLoader():
    """
    Class to load each tile, neighboring tiles, segmentation and neighboring segmentation.

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

        # Initialize attributes to load tile data and neighbors data.
        self.data = None
        self.data_neighbors = None
        self.data_segmentation = None
        self.data_neighbors_segmentation = None

    def load_tile(self):
        """
        Load the current tile.

        """
        if self.data is None:
            self.data = self._load_data(self.path)
        else:
            print('Tile already loaded.')

    def load_neighbors(self):
        """
        Load the current tile neighbors.
        """

        if self.data_neighbors is None:
            u = []
            for path_neighbor in self.neighbors_path:
                u.append(self._load_data(path_neighbor))
        else:
            print('Tile neighbors already loaded.')

        self.data_neighbors = u

    def load_segmentation(self):
        """
        Load the current tile cc.
        """

        if self.data_segmentation is None:
            apr = pyapr.APR()
            cc = pyapr.LongParticles()
            folder, filename = os.path.split(self.path)
            folder_seg = os.path.join(folder, 'segmentation')
            pyapr.io.read(os.path.join(folder_seg, filename[:-4] + '_segmentation.apr'), apr, cc)
            u = (apr, cc)
        else:
            print('Tile cc already loaded.')

        self.data_segmentation = u

    def load_neighbors_segmentation(self):
        """
        Load the current tile neighbors cc.
        """

        if self.data_neighbors_segmentation is None:
            u = []
            for path_neighbor in self.neighbors_path:
                apr = pyapr.APR()
                cc = pyapr.LongParticles()
                folder, filename = os.path.split(path_neighbor)
                folder_seg = os.path.join(folder, 'segmentation')
                pyapr.io.read(os.path.join(folder_seg, filename[:-4] + '_segmentation.apr'), apr, cc)
                u.append(apr, cc)
        else:
            print('Tile neighbors already loaded.')

        self.data_neighbors_segmentation = u

    def _load_data(self, path):
        """
        Load the current tile.

        """
        if self.type == 'tiff2D':
            u = self._load_sequence(path)
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

    @staticmethod
    def _load_sequence(path):
        """
        Load a sequence of images in a folder and return it as a 3D array.

        """
        files = glob(os.path.join(path, '*tif'))
        n_files = len(files)

        files_sorted = list(range(n_files))
        n_max = 0
        for i, pathname in enumerate(files):
            number_search = re.search('CHN00_PLN(\d+).tif', pathname)
            if number_search:
                n = int(number_search.group(1))
                files_sorted[n] = pathname
                if n > n_max:
                    n_max = n

        files_sorted = files_sorted[:n_max]
        n_files = len(files_sorted)

        u = imread(files_sorted[0])
        v = np.empty((n_files, *u.shape), dtype='uint16')
        v[0] = u
        files_sorted.pop(0)
        with alive_bar(n_files, force_tty=True, title='Loading sequence') as bar:
            for i, f in enumerate(files_sorted):
                v[i + 1] = imread(f)
                bar()

        return v
