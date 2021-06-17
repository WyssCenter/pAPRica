"""
Module containing classes and functions relative to data loading.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from glob import glob
import os
from skimage.io import imread
import numpy as np
import pipapr
import pyapr
import re
from alive_progress import alive_bar


class tileLoader():
    """
    Class to load each tile, neighboring tiles, segmentation and neighboring segmentation.

    Tile post processing is done on APR data, so if the input data is tiff it is first converted.
    """
    def __init__(self, path, row, col, ftype, neighbors, neighbors_path, overlap, frame_size, folder_root):

        self.path = path
        self.row = row
        self.col = col
        self.type = ftype
        self.neighbors = neighbors
        self.neighbors_path = neighbors_path
        self.overlap = overlap
        self.frame_size = frame_size
        self.folder_root = folder_root

        # Initialize attributes to load tile data
        self.data = None                    # Pixel data
        self.apr = None                     # APR tree
        self.parts = None                   # Particles
        self.parts_cc = None                # Connected component

        # Initialize attributes to load neighbors data
        self.data_neighbors = None
        self.apr_neighbors = None
        self.parts_neighbors = None
        self.parts_cc_neighbors = None

    def load_tile(self):
        """
        Load the current tile.

        """
        if self.data is None:
            if self.type == 'apr':
                self.apr, self.parts = self._load_data(self.path)
            else:
                self.data = self._load_data(self.path)
        else:
            print('Tile already loaded.')

    def load_neighbors(self):
        """
        Load the current tile neighbors.
        """
        if self.data_neighbors is None:
            if self.type == 'apr':
                aprs = []
                partss = []
                for path_neighbor in self.neighbors_path:
                    apr, parts = self._load_data(path_neighbor)
                    aprs.append(apr)
                    partss.append(parts)
                self.apr_neighbors = aprs
                self.parts_neighbors = partss
            else:
                u = []
                for path_neighbor in self.neighbors_path:
                    u.append(self._load_data(path_neighbor))
                self.data_neighbors = u
        else:
            print('Tile neighbors already loaded.')

    def load_segmentation(self, load_tree=False):
        """
        Load the current tile cc.
        """

        if self.parts_cc is None:
            cc = pyapr.LongParticles()
            aprfile = pyapr.io.APRFile()
            aprfile.set_read_write_tree(True)
            aprfile.open(self.path, 'READ')
            if load_tree:
                apr = pyapr.APR()
                aprfile.read_apr(apr, t=0, channel_name='t')
                self.apr = apr
            aprfile.read_particles(self.apr, 'segmentation cc', cc, t=0)
            aprfile.close()
            self.parts_cc = cc
        else:
            print('Tile cc already loaded.')

    def load_neighbors_segmentation(self, load_tree=False):
        """
        Load the current tile neighbors cc.
        """
        if self.data_neighbors is None:
            if self.type == 'apr':
                aprs = []
                ccs = []
                for i, path_neighbor in enumerate(self.neighbors_path):
                    if not load_tree:
                        apr = self.apr_neighbors[i]
                    cc = pyapr.LongParticles()
                    aprfile = pyapr.io.APRFile()
                    aprfile.set_read_write_tree(True)
                    aprfile.open(path_neighbor, 'READ')
                    if load_tree:
                        apr = pyapr.APR()
                        aprfile.read_apr(apr, t=0, channel_name='t')
                    aprfile.read_particles(apr, 'segmentation cc', cc, t=0)
                    aprfile.close()
                    aprs.append(apr)
                    ccs.append(cc)
                self.apr_neighbors = aprs
                self.parts_neighbors = ccs

            else:
                u = []
                for path_neighbor in self.neighbors_path:
                    u.append(self._load_data(path_neighbor))
                self.data_neighbors = u
        else:
            print('Tile neighbors already loaded.')

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
        elif self.type == 'raw':
            u = self._load_raw(path)
        else:
            raise TypeError('Error: image type {} not supported.'.format(self.type))

        return u

    def _load_raw(self, path):
        """
        Load a raw file (binary) using numpy.

        """
        u = np.fromfile(path, dtype='uint16', count=-1)
        return u.reshape((-1, self.frame_size, self.frame_size))

    def view_tile(self, **kwargs):
        """
        Display tile using napari.

        """
        if self.apr is None:
            self.load_tile()
        pipapr.viewer.display_apr(self.apr, self.parts)

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
