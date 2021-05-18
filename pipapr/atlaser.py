"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from skimage.io import imread, imsave
from pipapr.stitcher import tileMerger
import numpy as np
import os
from pathlib import Path

class tileAtlaser():

    def __init__(self,
                 original_pixel_size: (np.array, list),
                 merger: (tileMerger, None) = None,
                 downsample = None):

        if merger is not None:
            # Instantiating with a merger object for atlasing
            self.downsample = merger.downsample
            self.level_delta = merger.level_delta
            self.merged_data = merger.merged_data
            self.merger = merger
        else:
            # Instantiating without a merger for loading an atlas
            if downsample is None:
                raise ValueError('Error: instantiating tileAtlaser without a merger requires to pass the downsampling that was used.')
            self.downsample = downsample

        self.pixel_size_registered_atlas = np.array([25, 25, 25])
        self.pixel_size_data = np.array(original_pixel_size) # Z Y X

        self.atlas = None

    def load_atlas(self, path):
        """
        Function to load a previously computed atlas.

        Parameters
        ----------
        path: (str) path to the registered atlas file.

        Returns
        -------
        None
        """

        self.atlas = imread(path)
        self.atlas = np.swapaxes(self.atlas, 0, 1)
        self.z_downsample = self.pixel_size_registered_atlas[0] / self.pixel_size_data[0]
        self.y_downsample = self.pixel_size_registered_atlas[1] / self.pixel_size_data[1]
        self.x_downsample = self.pixel_size_registered_atlas[2] / self.pixel_size_data[2]

    def register_to_atlas(self,
                          output_dir='./',
                          orientation='spr',
                          merged_data_filename='merged_data.tif',
                          **kwargs):
        """
        Function to compute the registration to the Atlas. It is just a wrapper to call brainreg.

        Parameters
        ----------
        output_dir: (str) output directory to save atlas
        orientation: (str) orientation of the input data with respect to the origin in Z,Y,X order. E.g. 'spr' means
                        superior (so going from origin to z = zlim we go from superior to inferior), posterior
                        (so going from origin to y = ylim we go from posterior to anterior part) and right (so going
                         from origin to x = xlim we go from right to left part)
        merged_data_filename: (str) named of the merged array (Brainreg reads data from files so we need to save
                                the merged volume beforehand.
        kwargs: (dict) dictionnary with keys as brainreg options and values as parameters (see here:
                https://docs.brainglobe.info/brainreg/user-guide/parameters)

        Returns
        -------
        None
        """

        # Create directories if they do not exist
        atlas_dir = os.path.join(output_dir, 'atlas')
        Path(atlas_dir).mkdir(parents=True, exist_ok=True)

        path_merged_data = os.path.join(output_dir, merged_data_filename)
        imsave(path_merged_data, self.merged_data)
        command = 'brainreg {} {} -v {} {} {} --orientation {}'.format('"' + path_merged_data + '"',
                                                            '"' + atlas_dir + '"',
                                                            self.pixel_size[0],
                                                            self.pixel_size[1],
                                                            self.pixel_size[2],
                                                            orientation)
        for key, value in kwargs.items():
            command += ' --{} {}'.format(key, value)

        # Execute brainreg
        os.system(command)

        self.atlas = self.load_atlas(os.path.join(atlas_dir, 'registered_atlas.tif'))

    def get_cells_id(self, cells):

        ids = self.atlas[np.floor(cells.cells[:, 0]/self.z_downsample).astype('uint64'),
                        np.floor(cells.cells[:, 1]/self.y_downsample).astype('uint64'),
                        np.floor(cells.cells[:, 2]/self.x_downsample).astype('uint64')]

        return ids

    def get_loc_id(self, x, y, z):
        """
        Return the ID (brain region) of a given position (typically to retrieve cell position in the brain).

        Parameters
        ----------
        x: (int) x position
        y: (int) y position
        z: (int) z position

        Returns
        -------
        ID at the queried position.
        """

        return self.atlas[int(z / self.z_downsample), int(y / self.y_downsample), int(x / self.x_downsample)]
