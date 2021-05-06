"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from glob import glob
import os
from skimage.io import imread
import numpy as np
import pyapr
import napari
from napari.layers import Image, Labels


def apr_to_napari_Image(apr: pyapr.APR,
                        parts: (pyapr.ShortParticles, pyapr.FloatParticles),
                        mode: str = 'constant',
                        level_delta: int = 0,
                        **kwargs):
    """
    Construct a napari 'Image' layer from an APR. Pixel values are reconstructed on the fly via the APRSlicer class.

    Parameters
    ----------
    apr : pyapr.APR
        Input APR data structure
    parts : pyapr.FloatParticles or pyapr.ShortParticles
        Input particle intensities
    mode: str
        Interpolation mode to reconstruct pixel values. Supported values are
            constant:   piecewise constant interpolation
            smooth:     smooth interpolation (via level-adaptive separable smoothing). Note: significantly slower than constant.
            level:      interpolate the particle levels to the pixels
        (default: constant)
    level_delta: int
        Sets the resolution of the reconstruction. The size of the image domain is multiplied by a factor of 2**level_delta.
        Thus, a value of 0 corresponds to the original pixel image resolution, -1 halves the resolution and +1 doubles it.
        (default: 0)

    Returns
    -------
    out : napari.layers.Image
        An Image layer of the APR that can be viewed in napari.
    """
    if 'contrast_limits' in kwargs:
        contrast_limits = kwargs.get('contrast_limits')
        del kwargs['contrast_limits']
    else:
        cmin = apr.level_min() if mode == 'level' else parts.min()
        cmax = apr.level_max() if mode == 'level' else parts.max()
        contrast_limits = [cmin, cmax]
    return Image(data=pyapr.data_containers.APRSlicer(apr, parts, mode=mode, level_delta=level_delta),
                 rgb=False, multiscale=False, contrast_limits=contrast_limits, **kwargs)


def apr_to_napari_Labels(apr: pyapr.APR,
                        parts: pyapr.ShortParticles,
                        mode: str = 'constant',
                        level_delta: int = 0,
                        **kwargs):
    """
    Construct a napari 'Layers' layer from an APR. Pixel values are reconstructed on the fly via the APRSlicer class.

    Parameters
    ----------
    apr : pyapr.APR
        Input APR data structure
    parts : pyapr.FloatParticles or pyapr.ShortParticles
        Input particle intensities
    mode: str
        Interpolation mode to reconstruct pixel values. Supported values are
            constant:   piecewise constant interpolation
            smooth:     smooth interpolation (via level-adaptive separable smoothing). Note: significantly slower than constant.
            level:      interpolate the particle levels to the pixels
        (default: constant)
    level_delta: int
        Sets the resolution of the reconstruction. The size of the image domain is multiplied by a factor of 2**level_delta.
        Thus, a value of 0 corresponds to the original pixel image resolution, -1 halves the resolution and +1 doubles it.
        (default: 0)

    Returns
    -------
    out : napari.layers.Image
        A Labels layer of the APR that can be viewed in napari.
    """
    if 'contrast_limits' in kwargs:
        del kwargs['contrast_limits']

    return Labels(data=pyapr.data_containers.APRSlicer(apr, parts, mode=mode, level_delta=level_delta, tree_mode='max'),
                  multiscale=False, **kwargs)


def display_layers(layers):
    with napari.gui_qt():
        viewer = napari.Viewer()
        for layer in layers:
            viewer.add_layer(layer)
    return viewer


def display_segmentation(apr, parts, mask):
    """
    This function displays an image and its associated segmentation map. It uses napari to lazily generate the pixel
    data from APR on the fly.

    Parameters
    ----------
    apr: (APR) apr object
    parts: (ParticleData) particle object representing the image
    mask: (ParticleData) particle object reprenting the segmentation mask/connected component

    Returns
    -------

    """
    image_nap = apr_to_napari_Image(apr, parts, name='APR')
    mask_nap = apr_to_napari_Labels(apr, mask, name='Segmentation')

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_layer(image_nap)
        viewer.add_layer(mask_nap)


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
        df = self.tgraph.database
        path = df[(df['row'] == row) & (df['col'] == col)]['path'].values[0]
        apr = pyapr.APR()
        parts = pyapr.LongParticles()
        folder, filename = os.path.split(path)
        folder_seg = os.path.join(folder, 'segmentation')
        pyapr.io.read(os.path.join(folder_seg, filename[:-4] + '_segmentation.apr'), apr, parts)
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
        df = self.tgraph.database
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
            u = imread(path)
            return self._get_apr(u)
        elif self.tiles.type == 'apr':
            apr = pyapr.APR()
            parts = pyapr.ShortParticles()
            pyapr.io.read(path, apr, parts)
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