"""
Submodule containing classes and functions relative to data **loading**.

Usually, tileLoader objects are instantiated directly by iterating over the parser. Alternatively, they can be
instantiated directly by the constructor by calling *tile = paprica.loader.tileLoader()*.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import os
import shutil
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pyapr
from skimage.io import imread
from tqdm import tqdm

import paprica


def tile_from_apr(apr, parts):
    """
    Function to generate a *tile* object from an APR object.

    Parameters
    ----------
    apr: pyapr.APR
        APR to generate a tile from
    parts: pyapr.ParticleData
        ParticleData to generate a tile from

    Returns
    -------
    tile: tileLoader
        *tile* containing the given APR
    """

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
    """
    Function to generate a *tile* object from the path of an APR object.

    Parameters
    ----------
    path: string
        path to the stored APR object

    Returns
    -------
    tile: tileLoader
        *tile* containing the given APR
    """
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

    """
    def __init__(self, path, row, col, ftype, neighbors, neighbors_tot, neighbors_path, frame_size, folder_root,
                 channel):
        """
        Constructor of tileLoader object.

        Parameters
        ----------
        path: string
            path to the tile (APR and tiff3D) or the folder containing the frames (tiff2D)
        row: int
            vertical position of the tile (for multi-tile acquisition)
        col: int
            horizontal position of the tile (for multi-tile acquisition)
        ftype: str
            tile file type ('apr', 'tiff3D', 'colm', 'clearscope')
        neighbors: list
            neighbors list containing the neighbors position [row, col] of only the EAST and SOUTH
            neighbors to avoid the redundancy computation when stitching. For example, below the tile [0, 1] is
            represented by an 'o' while other tile are represented by an 'x'::

                            x --- o --- x --- x
                            |     |     |     |
                            x --- x --- x --- x
                            |     |     |     |
                            x --- x --- x --- x
                            |     |     |     |
                            x --- x --- x --- x

            in this case neighbors = [[0, 2], [1, 1]]

        neighbors_tot: list
            neighbors list containing all the neighbors position [row, col]. For example, below the
            tile [0, 1] is represented by an 'o' while other tile are represented by an 'x'::

                            x --- o --- x --- x
                            |     |     |     |
                            x --- x --- x --- x
                            |     |     |     |
                            x --- x --- x --- x
                            |     |     |     |
                            x --- x --- x --- x

            in this case neighbors_tot = [[0, 0], [0, 2], [1, 1]]
        neighbors_path: list
            path of the neighbors whose coordinates are stored in neighbors
        frame_size: int
            camera frame size (only square sensors are supported for now).
        folder_root: str
            root folder where everything should be saved.
        channel: int
            fluorescence channel for multi-channel acquisition. This is used to load the right data in the
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
        self.is_loaded = False

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

        Returns
        -------
        None
        """
        if self.type == 'apr':
            if self.apr is None:
                self.apr, self.parts = self._load_data(self.path)
        else:
            if self.data is None:
                self.data = self._load_data(self.path)
        self.is_loaded = True

    def lazy_load_tile(self, level_delta=0):
        """
        Load the tile lazily at the given resolution.

        Parameters
        ----------
        level_delta: int
            parameter controlling the resolution at which the APR will be read lazily

        Returns
        -------
        None
        """

        if self.type != 'apr':
            raise TypeError('Error: lazy loading is only supported for APR data.')

        self.lazy_data = pyapr.reconstruction.LazySlicer(self.path, level_delta=level_delta, parts_name='particles',
                                                          tree_parts_name='particles')
        self.is_loaded = True

    def load_neighbors(self):
        """
        Load the current tile neighbors if not already loaded.

        Returns
        -------
        None
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

        Returns
        -------
        None
        """
        if self.parts_cc is None:
            self.parts_cc = pyapr.io.read_particles(self.path, parts_name='segmentation cc')
            if load_tree:
                self.apr = pyapr.io.read_apr(self.path)
        else:
            print('Tile cc already loaded.')

    def lazy_load_segmentation(self, level_delta=0):
        """
        Load the parts_cc lazily at the given resolution.

        Parameters
        ----------
        level_delta: int
            parameter controlling the resolution at which the APR will be read lazily

        Returns
        -------
        None
        """
        if self.type != 'apr':
            raise TypeError('Error: lazy loading is only supported for APR data.')

        self.lazy_segmentation = pyapr.reconstruction.LazySlicer(self.path, level_delta=level_delta,
                                                                 parts_name='segmentation cc',
                                                                 tree_parts_name='segmentation cc')

    def load_neighbors_segmentation(self, load_tree=False):
        """
        Load the current tile neighbors connected component (cc) if not already loaded.

        Returns
        -------
        None
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

    def view_tile(self, **kwargs):
        """
        Display tile using napari.

        Returns
        -------
        None
        """
        if self.apr is None:
            self.load_tile()
        paprica.viewer.display_apr(self.apr, self.parts, **kwargs)

    def plot_particles_size_distribution(self):
        """
        Plot the particle size distribution of the tile.

        Returns
        -------
        None
        """
        if self.type != 'apr':
            raise TypeError('Error: particles distributoin can only be computed for APR.')

        if not self.is_loaded:
            self.load_tile()

        it = self.apr.iterator()
        nparts = []
        for level in range(self.apr.level_min(), self.apr.level_max()):
            nparts.append(it.total_number_particles(level + 1) - it.total_number_particles(level))
        x = np.array([2 ** x for x in range(self.apr.level_max() - self.apr.level_min() - 1, -1, -1)])
        plt.figure()
        plt.bar(x[2:], nparts[2:] / np.sum(nparts), width=np.diff(x[:-1]) / 10, log=True, align="center")
        plt.xscale('log')
        plt.xlabel('Particle size [pixel]', fontsize=14)
        plt.ylabel('Normalized particle distribution', fontsize=14)
        plt.xticks(x[2:], x[2:])
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.title('CR = {:0.2f}'.format(self.apr.computational_ratio()))
        plt.tight_layout()

    def _compute_segmentation_cc_tree_particles(self):

        # Load APR file
        self.apr, self.parts = self._load_data(self.path)

        # Compute tree particles
        apr, parts = pyapr.io.read(self.path, parts_name='segmentation cc')
        tree_parts = pyapr.tree.fill_tree_max(apr, parts)

        # Save back data
        pyapr.io.write_particles(self.path, tree_parts, t=0, channel_name='t', parts_name='segmentation cc', tree=True, append=True)

    def _load_data(self, path):
        """
        Load data at given path.

        Parameters
        ----------
        path: string
            path to the data to be loaded.

        Returns
        -------
        u: array_like
            numpy array containing the data.
        """
        if self.type == 'colm':
            u = self._load_colm(path)
        elif self.type == 'clearscope':
            u = self._load_clearscope(path)
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
        path: string
            path to the data to be loaded.

        Returns
        -------
        u: array_like
            numpy array containing the data.
        """
        u = np.fromfile(path, dtype='uint16', count=-1)
        return u.reshape((-1, self.frame_size, self.frame_size))

    def _load_colm(self, path):
        """
        Load a sequence of images in a folder and return it as a 3D array.

        Parameters
        ----------
        path: string
            path to folder where the data should be loaded.

        Returns
        -------
        v: array_like
            numpy array containing the data.
        """
        files_sorted = sorted(glob(os.path.join(path, '*CHN0' + str(self.channel) + '_*tif')))
        n_files = len(files_sorted)
        v = np.empty((n_files, self.frame_size, self.frame_size), dtype='uint16')
        for i, f in enumerate(tqdm(files_sorted, desc='Loading sequence', leave=False)):
            v[i] = imread(f)
        return v

    def _load_clearscope(self, path):
        """
        Load a sequence of images in a folder and return it as a 3D array.

        Parameters
        ----------
        path: string
            path to folder where the data should be loaded.

        Returns
        -------
        v: array_like
            numpy array containing the data.
        """
        files_sorted = sorted(glob(os.path.join(path, '*')))
        n_files = len(files_sorted)
        v = np.empty((n_files, self.frame_size, self.frame_size), dtype='uint16')
        for i, f in enumerate(tqdm(files_sorted, desc='Loading sequence', leave=False)):
            v[i] = imread(f)

        return v

    # def _load_mesospim(self, path):
    #     """
    #     Load MESOSPIM data and return it as a 3D array.
    #
    #     Parameters
    #     ----------
    #     path: string
    #         path to folder where the data should be loaded.
    #
    #     Returns
    #     -------
    #     v: array_like
    #         numpy array containing the data.
    #     """
    #     if path.endswith('.raw'):
    #         return self._load_raw(path)
    #     elif path.endswith('.tiff') or path.endswith('.tif'):
    #         return imread(path)

    def _erase_from_disk(self):
        """
        Delete tile from disk, use with caution!

        Returns
        -------
        None
        """

        if self.type == 'apr':
            os.remove(self.path)
        else:
            shutil.rmtree(self.path)


