"""
Submodule containing classes and functions relative to data **loading**.

Usually, tileLoader objects are instantiated directly by iterating over the parser. Alternatively, they can be
instantiated directly by the constructor by calling *tile = pipapr.loader.tileLoader()*.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from glob import glob
import os
from skimage.io import imread
import numpy as np
import pipapr
import pyapr
from tqdm import tqdm

def tile_from_apr(apr, parts):

    tile = tileLoader(path=None,
                      row=None,
                      col=None,
                      ftype='apr',
                      neighbors=None,
                      neighbors_tot=None,
                      neighbors_path=None,
                      frame_size=2048,
                      folder_root=None,
                      channel=None)

    tile.apr = apr
    tile.parts = parts

    return tile

def tile_from_path(path):

    return tileLoader(path=path,
                      row=1,
                      col=1,
                      ftype='apr',
                      neighbors=None,
                      neighbors_tot=None,
                      neighbors_path=None,
                      frame_size=2048,
                      folder_root=os.path.basename(path),
                      channel=None)

class tileLoader():
    """
    Class to load each tile, neighboring tiles, segmentation and neighboring segmentation.

    Tile post processing is done on APR data, so if the input data is tiff it is first converted.
    """
    def __init__(self, path, row, col, ftype, neighbors, neighbors_tot, neighbors_path, frame_size, folder_root, channel):
        """
        Constructor of tileLoader object.

        Parameters
        ----------
        path: (str) path to the tile (APR and tiff3D) or the folder containing the frames (tiff2D)
        row: (int) vertical position of the tile (for multi-tile acquisition)
        col: (int) horizontal position of the tile (for multi-tile acquisition)
        ftype: (str) tile file type ('apr', 'tiff3D', 'tiff2D')
        neighbors: (list) neighbors list containing the neighbors position [row, col] of only the EAST and SOUTH
                    neighbors to avoid the redundancy computation when stitching. For example, below the tile [0, 1] is
                    represented by an 'o' while other tile are represented by an 'x':

                            x --- o --- x --- x
                            |     |     |     |
                            x --- x --- x --- x
                            |     |     |     |
                            x --- x --- x --- x
                            |     |     |     |
                            x --- x --- x --- x

                    in this case neighbors = [[0, 2], [1, 1]]

        neighbors_tot: (list) neighbors list containing all the neighbors position [row, col]. For example, below the
                        tile [0, 1] is represented by an 'o' while other tile are represented by an 'x':

                            x --- o --- x --- x
                            |     |     |     |
                            x --- x --- x --- x
                            |     |     |     |
                            x --- x --- x --- x
                            |     |     |     |
                            x --- x --- x --- x

                    in this case neighbors_tot = [[0, 0], [0, 2], [1, 1]]
        neighbors_path: (list) path of the neighbors whose coordinates are stored in neighbors
        frame_size: (int) camera frame size (only square sensors are supported for now).
        folder_root: (str) root folder where everything should be saved.
        channel: (int) fluorescence channel for multi-channel acquisition. This is used to load the right data in the
                        case of COLM acquisition where all the channel are saved in the same folder as tiff2D.
        """

        self.path = path
        self.row = row
        self.col = col
        self.type = ftype
        self.neighbors = neighbors
        self.neighbors_tot = neighbors_tot
        self.neighbors_path = neighbors_path
        self.frame_size = frame_size
        self.folder_root = folder_root
        self.channel = channel

        # Initialize attributes to load tile data
        self.data = None                    # Pixel data
        self.apr = None                     # APR tree
        self.parts = None                   # Particles
        self.parts_cc = None                # Connected component
        self.lazy_data = None               # Lazy reconstructed data

        # Initialize attributes to load neighbors data
        self.data_neighbors = None
        self.apr_neighbors = None
        self.parts_neighbors = None
        self.parts_cc_neighbors = None

    def load_tile(self):
        """
        Load the current tile if not already loaded.

        """
        if self.type == 'apr':
            if self.apr is None:
                self.apr, self.parts = self._load_data(self.path)
        else:
            if self.data is None:
                self.data = self._load_data(self.path)

    def lazy_load_tile(self, level_delta=0):

        if self.type != 'apr':
            raise TypeError('Error: lazy loading is only supported for APR data.')

        self.lazy_data = pyapr.data_containers.LazySlicer(self.path, level_delta=level_delta)

    def load_neighbors(self):
        """
        Load the current tile neighbors if not already loaded.

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
        Load the current tile connected component (cc) if not already loaded.

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
        Load the current tile neighbors connected component (cc) if not already loaded.

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
        Load data at given path.

        Parameters
        ----------
        path: (str) path to the data to be loaded.

        Returns
        -------
        u: (array) numpy array containing the data.
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
        Load raw data at given path.

        Parameters
        ----------
        path: (str) path to the data to be loaded.

        Returns
        -------
        u: (array) numpy array containing the data.
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

    def _load_sequence(self, path):
        """
        Load a sequence of images in a folder and return it as a 3D array.

        Parameters
        ----------
        path: (str) path to folder where the data should be loaded.

        Returns
        -------
        v: (array) numpy array containing the data.
        """
        files_sorted = sorted(glob(os.path.join(path, '*CHN0' + str(self.channel) + '_*tif')))
        n_files = len(files_sorted)
        #
        # files_sorted = list(range(n_files))
        # n_max = 0
        # for i, pathname in enumerate(files):
        #     number_search = re.search('CHN0' + str(self.channel) + '_PLN(\d+).tif', pathname)
        #     if number_search:
        #         n = int(number_search.group(1))
        #         files_sorted[n] = pathname
        #         if n > n_max:
        #             n_max = n
        #
        # files_sorted = files_sorted[:n_max]
        # n_files = len(files_sorted)

        u = imread(files_sorted[0])
        v = np.empty((n_files, *u.shape), dtype='uint16')
        v[0] = u
        files_sorted.pop(0)
        for i, f in enumerate(tqdm(files_sorted, desc='Loading sequence', leave=False)):
            v[i + 1] = imread(f)

        return v
