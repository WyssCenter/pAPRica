"""
Module containing classes and functions relative to Viewing.


By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import pyapr
from matplotlib.colors import LogNorm
from napari.layers import Image, Labels, Points
from skimage.color import hsv2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.io import imread
from skimage.transform import resize

import paprica


def display_apr_from_path(path, **kwargs):
    """
    Display an APR using Napari from a filepath.

    Parameters
    ----------
    path: string
        path to APR to be displayed
    kwargs: dict
        optional parameters for Napari

    Returns
    -------
    None
    """
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    pyapr.io.read(path, apr, parts)
    layer = apr_to_napari_Image(apr, parts)
    display_layers_pyramidal([layer], level_delta=0, **kwargs)


def display_apr(apr, parts, **kwargs):
    """
    Display an APR using Napari from previously loaded data.

    Parameters
    ----------
    apr : pyapr.APR
        Input APR data structure
    parts : pyapr.FloatParticles, pyapr.ShortParticles
        Input particle intensities
    kwargs: dict
        optional parameters for Napari

    Returns
    -------
    None
    """
    l = apr_to_napari_Image(apr, parts, **kwargs)
    display_layers_pyramidal([l], level_delta=0)


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
    mode: str (default: 'constant')
        Interpolation mode to reconstruct pixel values. Supported values are
            constant:   piecewise constant interpolation
            smooth:     smooth interpolation (via level-adaptive separable smoothing). Note: significantly slower than constant.
            level:      interpolate the particle levels to the pixels
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
    if 'tree_mode' in kwargs:
        tree_mode = kwargs.get('tree_mode')
        del kwargs['tree_mode']
    else:
        tree_mode = 'mean'
    par = apr.get_parameters()
    return Image(data=pyapr.reconstruction.APRSlicer(apr, parts, mode=mode, level_delta=level_delta, tree_mode=tree_mode),
                 rgb=False, multiscale=False, contrast_limits=contrast_limits,
                 scale=[par.dz, par.dx, par.dy], **kwargs)


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
    mode: str (default: 'constant')
        Interpolation mode to reconstruct pixel values. Supported values are
            constant:   piecewise constant interpolation
            smooth:     smooth interpolation (via level-adaptive separable smoothing). Note: significantly slower than constant.
            level:      interpolate the particle levels to the pixels
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
    par = apr.get_parameters()
    return Labels(data=pyapr.reconstruction.APRSlicer(apr, parts, mode=mode, level_delta=level_delta, tree_mode='max'),
                  multiscale=False, scale=[par.dz, par.dx, par.dy], **kwargs)

# Define a callback that will take the value of the slider and the viewer
def resolution_callback(viewer, value):
    for l in viewer.layers:
        if isinstance(l.data, pyapr.reconstruction.APRSlicer):
            old_value = -l.data.patch.level_delta
            l.data.set_level_delta(-value)
            l.translate = l.translate/2**(value-old_value)
    viewer.dims.set_point(axis=0, value=viewer.dims.point[0] / 2 ** (value-old_value))
    viewer.status = str(value)
    viewer._update_layers()
    viewer.reset_view()


def display_layers(layers):
    """
    Display a list of layers using Napari.

    Parameters
    ----------
    layers: list[napari.Layer]
        list of layers to display

    Returns
    -------
    viewer: napari.Viewer
        napari viewer.
    """

    viewer = napari.Viewer()
    for layer in layers:
        viewer.add_layer(layer)

    napari.run()


    return viewer


def display_layers_pyramidal(layers, level_delta):
    """
    Display a list of layers using Napari.

    Parameters
    ----------
    layers: list[napari.Layer]
        list of layers to display

    Returns
    -------
    viewer: napari.Viewer
        napari viewer.
    """

    viewer = napari.Viewer()
    for layer in layers:
        viewer.add_layer(layer)

    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSlider


    my_slider = QSlider(Qt.Horizontal)
    my_slider.setMinimum(0)
    layers_apr = [l for l in layers if isinstance(l.data, pyapr.reconstruction.APRSlicer)]
    l_max = np.min([l.data.apr.level_max() for l in layers_apr])
    l_min = 5 if l_max > 5 else 1
    my_slider.setMaximum(l_max-l_min)
    my_slider.setSingleStep(1)
    my_slider.setValue(-level_delta)

    # Connect your slider to your callback function
    my_slider.valueChanged[int].connect(
        lambda value=my_slider: resolution_callback(viewer, value)
    )
    viewer.window.add_dock_widget(my_slider, name='Downsampling', area='left')

    napari.run()


    return viewer


def display_segmentation(apr, parts, mask, pyramidal=True, **kwargs):
    """
    This function displays an image and its associated segmentation map. It uses napari to lazily generate the pixel
    data from APR on the fly.

    Parameters
    ----------
    apr: pyapr.APR
        apr object
    parts: pyapr.ParticleData
        particle object representing the image
    mask: pyapr.ParticleData
        particle object representing the segmentation mask/connected component

    Returns
    -------
    None
    """
    layers = []
    layers.append(apr_to_napari_Image(apr, parts, name='APR', **kwargs))
    layers.append(apr_to_napari_Labels(apr, mask, name='Segmentation', opacity=0.3, **kwargs))
    if pyramidal:
        display_layers_pyramidal(layers, level_delta=0)
    else:
        display_layers(layers)


def display_heatmap(heatmap, atlas=None, data=None, log=False):
    """
    Display a heatmap (e.g. cell density) that can be overlaid on intensity data and atlas.

    Parameters
    ----------
    heatmap: ndarray
        array containing the heatmap to be displayed
    atlas: ndarray
        array containing the atlas which will be automatically scaled to the heatmap
    data: ndarray
        array containing the data.
    log: bool
        plot in logscale (only used for 2D).

    Returns
    -------
    None
    """
    # If u is 2D then use matplotlib so we have a scale bar
    if heatmap.ndim == 2:
        fig, ax = plt.subplots()
        if log:
            h = ax.imshow(heatmap, norm=LogNorm(), cmap='jet')
        else:
            h = ax.imshow(heatmap, cmap='jet')
        cbar = fig.colorbar(h, ax=ax)
        cbar.set_label('Number of detected cells')
        ax.set_xticks([])
        ax.set_yticks([])
    # If u is 3D then use napari but no colorbar for now
    elif heatmap.ndim == 3:
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(heatmap, colormap='inferno', name='Heatmap', blending='additive', opacity=0.7)
            if atlas is not None:
                viewer.add_labels(atlas, name='Atlas regions', opacity=0.7)
            if data is not None:
                viewer.add_image(data, name='Intensity data', blending='additive',
                                 scale=np.array(heatmap.shape)/np.array(data.shape), opacity=0.7)


def compare_stitching(stitcher1, stitcher2, loc=None, n_proj=0, dim=0, downsample=2, color=False, rel_map=False):
    """
    Compare two stitching at a given position `loc` for a given dimension `dim`.

    Parameters
    ----------
    stitcher1: tileStitcher
        stitcher object 1
    stitcher2: tileStitcher
        stitcher object 2
    loc: int
        position in the given dimension
    dim: int
        dimension to use for comparison
    n_proj: int
        number of plane to perform the max-projection
    downsample: int
        downsampling factor for the reconstruction
    color: bool
        option to display in color
    rel_map: bool
        overlay reliability map on the reconstructed data

    Returns
    -------
    None
    """
    u1 = stitcher1.reconstruct_slice(loc=loc, n_proj=n_proj, dim=dim, downsample=downsample, color=color, plot=False)
    u2 = stitcher2.reconstruct_slice(loc=loc, n_proj=n_proj, dim=dim, downsample=downsample, color=color, plot=False)

    if color:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        data_to_display = np.ones_like(u1, dtype='uint8')
        for i in range(2):
            tmp = np.log(u1[:, :, i] + 200)
            vmin, vmax = np.percentile(tmp[tmp > np.log(1 + 200)], (1, 99.9))
            data_to_display[:, :, i] = rescale_intensity(tmp, in_range=(vmin, vmax), out_range='uint8')
        ax[0].imshow(data_to_display)
        data_to_display = np.ones_like(u2, dtype='uint8')
        for i in range(2):
            tmp = np.log(u2[:, :, i] + 200)
            vmin, vmax = np.percentile(tmp[tmp > np.log(1 + 200)], (1, 99.9))
            data_to_display[:, :, i] = rescale_intensity(tmp, in_range=(vmin, vmax), out_range='uint8')
        ax[1].imshow(data_to_display)
    else:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(np.log(u1), cmap='gray')
        if rel_map:
            try:
                rel_map = resize(np.mean(stitcher1.plot_stitching_info(), axis=0), u1.shape, order=1)
                ax[0].imshow(rel_map, cmap='turbo', alpha=0.5)
            except:
                pass
        ax[1].imshow(np.log(u2), cmap='gray')
        if rel_map:
            try:
                rel_map = resize(np.mean(stitcher2.plot_stitching_info(), axis=0), u1.shape, order=1)
                ax[1].imshow(rel_map, cmap='turbo', alpha=0.5)
            except:
                pass


def reconstruct_colored_projection(apr, parts, loc=None, dim=0, n_proj=0, downsample=1, threshold=None, plot=True):
    """
    Compare two stitching at a given position `loc` for a given dimension `dim`.

    Parameters
    ----------
    apr: pyapr.APR
        apr tree object
    parts: pyapr.ParticleData
        apr particles
    loc: int
        position in the given dimension
    dim: int
        dimension to use for comparison
    n_proj: int
        number of plane to perform the max-projection
    downsample: int
        downsampling factor for the reconstruction
    color: bool
        option to display in color
    rel_map: bool
        overlay reliability map on the reconstructed data

    Returns
    -------
    None
    """

    level_delta = int(-np.sign(downsample) * np.log2(np.abs(downsample)))

    if loc is None:
        apr_shape = apr.shape()
        loc = int(apr_shape[dim] / 2)

    if loc > apr_shape[dim]:
        raise ValueError('Error: loc is too large ({}), maximum loc at this downsample is {}.'.format(loc, apr_shape[dim]))

    locf = min(loc+n_proj, apr_shape[dim])
    patch = pyapr.ReconPatch()
    if dim == 0:
        patch.z_begin = loc
        patch.z_end = locf
    if dim == 1:
        patch.y_begin = loc
        patch.y_end = locf
    if dim == 2:
        patch.x_begin = loc
        patch.x_end = locf

    data = pyapr.reconstruction.reconstruct_constant(apr, parts, patch=patch)

    V = data.max(axis=dim)
    S = np.ones_like(V) * 0.7
    if threshold is not None:
        S[V<threshold] = 0
    H = np.argmax(data, axis=dim)
    H = rescale_intensity(gaussian(H, sigma=5), out_range=np.float64)*0.66
    V = np.log(V + 200)
    vmin, vmax = np.percentile(V[V > np.log(100)], (1, 99.9))
    V = rescale_intensity(V, in_range=(vmin, vmax), out_range=np.float64)
    S = S * V
    rgb = hsv2rgb(np.dstack((H,S,V)))
    rescale_intensity(rgb, out_range='uint8')

    if plot:
        plt.figure()
        plt.imshow(rgb)

    return rgb


class tileViewer():
    """
    Class to display the registration and segmentation using Napari.

    """
    def __init__(self,
                 tiles,
                 database,
                 segmentation: bool=False,
                 cells=None,
                 atlaser=None):
        """

        Parameters
        ----------
        tiles: tileParser
            tileParser object containing the dataset to be displayed.
        database: (pd.Dataframe, string, tileStitcher)
            database containing the tile positions.
        segmentation: bool
            option to also display the segmentation (connected component) data.
        cells: ndarray
            cells center to be displayed.
        atlaser: tileAtlaser
            tileAtlaser object containing the Atlas to be displayed.
        """
        self.tiles = tiles

        if isinstance(database, paprica.stitcher.tileStitcher):
            self.database = database.database
        elif isinstance(database, pd.DataFrame):
            self.database = database
        elif isinstance(database, str):
            self.database = pd.read_csv(database)
        else:
            raise TypeError('Error: unknown type for database.')

        self.nrow = tiles.nrow
        self.ncol = tiles.ncol
        self.loaded_ind = []
        self.loaded_tiles = {}
        self.segmentation = segmentation
        self.loaded_segmentation = {}
        self.cells = cells
        self.atlaser = atlaser

    def get_layers_all_tiles(self, downsample=1, **kwargs):
        """
        Display all parsed tiles.

        Parameters
        ----------
        downsample: int
            downsampling parameter for APRSlicer (1: full resolution, 2: 2x downsampling, 4: 4x downsampling..etc)
        kwargs: dict
            dictionary passed to Napari for custom option

        Returns
        -------
        layers: list[napari.Layer]
            list of layers to be displayed by Napari
        """

        # Compute layers to be displayed by Napari
        layers = []

        # Convert downsample to level delta
        level_delta = int(-np.sign(downsample)*np.log2(np.abs(downsample)))

        for tile in self.tiles:
            # Load tile if not loaded, else use cached tile
            ind = np.ravel_multi_index((tile.row, tile.col), dims=(self.nrow, self.ncol))
            if self._is_tile_loaded(tile.row, tile.col):
                apr, parts = self.loaded_tiles[ind]
                if self.segmentation:
                    cc = self.loaded_segmentation[ind]
            else:
                tile.load_tile()
                apr, parts = tile.apr, tile.parts
                self.loaded_ind.append(ind)
                self.loaded_tiles[ind] = apr, parts
                if self.segmentation:
                    tile.load_segmentation()
                    cc = tile.parts_cc
                    self.loaded_segmentation[ind] = cc


            position = self._get_tile_position(tile.row, tile.col)
            if level_delta != 0:
                position = [x/downsample for x in position]
            layers.append(apr_to_napari_Image(apr, parts,
                                              mode='constant',
                                              name='Tile [{}, {}]'.format(tile.row, tile.col),
                                              translate=position,
                                              opacity=0.7,
                                              level_delta=level_delta,
                                              **kwargs))
            if self.segmentation:
                layers.append(apr_to_napari_Labels(apr, cc,
                                                   mode='constant',
                                                   name='Segmentation [{}, {}]'.format(tile.row, tile.col),
                                                   translate=position,
                                                   level_delta=level_delta,
                                                   opacity=0.7))
        if self.cells is not None:
            par = apr.get_parameters()
            layers.append(Points(self.cells, opacity=0.7, name='Cells center',
                                 scale=[par.dz/downsample, par.dx/downsample, par.dy/downsample]))

        if self.atlaser is not None:
            layers.append(Labels(self.atlaser.atlas, opacity=0.7, name='Atlas',
                                 scale=[self.atlaser.z_downsample/downsample,
                                        self.atlaser.y_downsample/downsample,
                                        self.atlaser.x_downsample/downsample]))

        return layers

    def display_all_tiles(self, pyramidal=True, downsample=1, color=False, **kwargs):
        """
        Display all parsed tiles.

        Parameters
        ----------
        pyramidal: bool
            option to have a slider that controls the displayed resolution
        downsample: int
            downsampling parameter for APRSlicer (1: full resolution, 2: 2x downsampling, 4: 4x downsampling..etc)
        kwargs: dict
            dictionary passed to Napari for custom option

        Returns
        -------
        None
        """

        # Compute layers to be displayed by Napari
        layers = []

        # Convert downsample to level delta
        level_delta = int(-np.sign(downsample)*np.log2(np.abs(downsample)))

        for tile in self.tiles:
            # Load tile if not loaded, else use cached tile
            ind = np.ravel_multi_index((tile.row, tile.col), dims=(self.nrow, self.ncol))
            if self._is_tile_loaded(tile.row, tile.col):
                apr, parts = self.loaded_tiles[ind]
                if self.segmentation:
                    cc = self.loaded_segmentation[ind]
            else:
                tile.load_tile()
                apr, parts = tile.apr, tile.parts
                self.loaded_ind.append(ind)
                self.loaded_tiles[ind] = apr, parts
                if self.segmentation:
                    tile.load_segmentation()
                    cc = tile.parts_cc
                    self.loaded_segmentation[ind] = cc

            position = self._get_tile_position(tile.row, tile.col)

            if color:
                blending = 'additive'
                if tile.col % 2:
                    if tile.row % 2:
                        cmap = 'red'
                    else:
                        cmap = 'green'
                else:
                    if tile.row % 2:
                        cmap = 'green'
                    else:
                        cmap = 'red'
            else:
                cmap = 'gray'
                blending = 'translucent'

            if level_delta != 0:
                position = [x/downsample for x in position]
            layers.append(apr_to_napari_Image(apr, parts,
                                              mode='constant',
                                              name='Tile [{}, {}]'.format(tile.row, tile.col),
                                              translate=position,
                                              opacity=0.7,
                                              level_delta=level_delta,
                                              **kwargs))
            if self.segmentation:
                layers.append(apr_to_napari_Labels(apr, cc,
                                                   mode='constant',
                                                   name='Segmentation [{}, {}]'.format(tile.row, tile.col),
                                                   translate=position,
                                                   level_delta=level_delta,
                                                   blending=blending,
                                                   opacity=0.7))
        if self.cells is not None:
            par = apr.get_parameters()
            layers.append(Points(self.cells, opacity=0.7, name='Cells center',
                                 scale=[par.dz/downsample, par.dx/downsample, par.dy/downsample]))

        if self.atlaser is not None:
            layers.append(Labels(self.atlaser.atlas, opacity=0.7, name='Atlas',
                                 scale=[self.atlaser.z_downsample/downsample,
                                        self.atlaser.y_downsample/downsample,
                                        self.atlaser.x_downsample/downsample]))

        # Display layers
        if pyramidal:
            display_layers_pyramidal(layers, level_delta)
        else:
            display_layers(layers)

    def display_tiles(self, coords, pyramidal=True, downsample=1, color=False, **kwargs):
        """
        Display tiles at position coords.

        Parameters
        ----------
        coords: list
            list of tuples (row, col) containing the tile coordinate to display.
        downsample: int
            downsampling parameter for APRSlicer (1: full resolution, 2: 2x downsampling, 4: 4x downsampling..etc)
        kwargs: dict
            dictionary passed to Napari for custom option
        color: bool
            option to display in color

        Returns
        -------
        None
        """

        # Compute layers to be displayed by Napari
        layers = []

        # Convert downsample to level delta
        level_delta = int(-np.sign(downsample) * np.log2(np.abs(downsample)))

        for tile in self.tiles:
            if (tile.row, tile.col) in coords:
                # Load tile if not loaded, else use cached tile
                ind = np.ravel_multi_index((tile.row, tile.col), dims=(self.nrow, self.ncol))
                if self._is_tile_loaded(tile.row, tile.col):
                    apr, parts = self.loaded_tiles[ind]
                    if self.segmentation:
                        cc = self.loaded_segmentation[ind]
                else:
                    tile.load_tile()
                    apr, parts = tile.apr, tile.parts
                    self.loaded_ind.append(ind)
                    self.loaded_tiles[ind] = apr, parts
                    if self.segmentation:
                        tile.load_segmentation()
                        cc = tile.parts_cc
                        self.loaded_segmentation[ind] = cc

                position = self._get_tile_position(tile.row, tile.col)
                if level_delta != 0:
                    position = [x / downsample for x in position]

                if color:
                    blending = 'additive'
                    if tile.col % 2:
                        if tile.row % 2:
                            cmap = 'red'
                        else:
                            cmap = 'green'
                    else:
                        if tile.row % 2:
                            cmap = 'green'
                        else:
                            cmap = 'red'
                else:
                    cmap = 'gray'
                    blending = 'translucent'

                layers.append(apr_to_napari_Image(apr, parts,
                                                  mode='constant',
                                                  name='Tile [{}, {}]'.format(tile.row, tile.col),
                                                  translate=position,
                                                  opacity=0.7,
                                                  level_delta=level_delta,
                                                  colormap=cmap,
                                                  blending=blending,
                                                  **kwargs))
                if self.segmentation:
                    layers.append(apr_to_napari_Labels(apr, cc,
                                                       mode='constant',
                                                       name='Segmentation [{}, {}]'.format(tile.row, tile.col),
                                                       translate=position,
                                                       level_delta=level_delta,
                                                       opacity=0.7))
        if self.cells is not None:
            par = apr.get_parameters()
            layers.append(Points(self.cells, opacity=0.7, name='Cells center',
                                 scale=[par.dz / downsample, par.dx / downsample, par.dy / downsample]))

        if self.atlaser is not None:
            layers.append(Labels(self.atlaser.atlas, opacity=0.7, name='Atlas',
                                 scale=[self.atlaser.z_downsample / downsample,
                                        self.atlaser.y_downsample / downsample,
                                        self.atlaser.x_downsample / downsample]))

        # Display layers
        if pyramidal:
            display_layers_pyramidal(layers, level_delta)
        else:
            display_layers(layers)

    def check_stitching(self, downsample=8, color=False, **kwargs):
        """
        Function to display the stitched dataset using napari.

        Parameters
        ----------
        downsample: int
            downsampling parameter for APRSlicer (1: full resolution, 2: 2x downsampling, 4: 4x downsampling..etc)
        color: bool
            option to display in color
        kwargs: dict
            dictionary passed to Napari for custom option

        Returns
        -------
        None
        """

        # Compute layers to be displayed by Napari
        layers = []

        # Convert downsample to level delta
        level_delta = int(-np.sign(downsample)*np.log2(np.abs(downsample)))

        for tile in self.tiles:
            tile.lazy_load_tile(level_delta=level_delta)
            position = self._get_tile_position(tile.row, tile.col)

            if level_delta != 0:
                position = [x/downsample for x in position]

            if color:
                blending = 'additive'
                if tile.col % 2:
                    if tile.row % 2:
                        cmap = 'red'
                    else:
                        cmap = 'green'
                else:
                    if tile.row % 2:
                        cmap = 'green'
                    else:
                        cmap = 'red'
            else:
                cmap = 'gray'
                blending = 'translucent'

            layers.append(Image(tile.lazy_data,
                              name='Tile [{}, {}]'.format(tile.row, tile.col),
                              translate=position,
                              opacity=0.7,
                              colormap=cmap,
                              blending=blending,
                              **kwargs))

        display_layers(layers)

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
        df = self.database
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

    def _get_tile_position(self, row, col):
        """
        Parse tile position in the database.

        """
        df = self.database
        tile_df = df[(df['row'] == row) & (df['col'] == col)]
        px = tile_df['ABS_H'].values[0]
        py = tile_df['ABS_V'].values[0]
        pz = tile_df['ABS_D'].values[0]

        return [pz, py, px]