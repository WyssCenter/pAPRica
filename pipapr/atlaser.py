"""
Submodule containing classes and functions relative to **atlasing**.

This submodule is essentially a wrapper to Brainreg (https://github.com/brainglobe/brainreg) for atlasing and
Allen Brain Atlas for ontology analysis. It contains many convenience method for manipulating data
(per region, per superpixel, etc.).

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pandas as pd
from skimage.io import imread, imsave
from skimage.filters import gaussian
import pipapr
import numpy as np
import os
from pathlib import Path
from allensdk.core.reference_space_cache import ReferenceSpaceCache

class tileAtlaser():
    """
    Class used for registering a dataset to the Atlas and do some post processing using the Atlas (e.g count cells
    per region).

    It can be instantiated using a tileMerger object (for registration using Brainreg) or directly with a
    previously registered Atlas.
    """

    def __init__(self,
                 original_pixel_size: (np.array, list),
                 downsample: int,
                 atlas=None,
                 merged_data=None):
        """
        Parameters
        ----------
        original_pixel_size: (np.array, list) pixel size in µm on the original data
        downsample: (int) downsampling used by APRSlicer to reconstruct the lower resolution pixel data used
                            for registration to the Atlas.
        atlas: (np.array, str) atlas data or path for loading the atlas data
        merger: (tileMerger) tileMerger object
        """

        self.downsample = downsample
        self.pixel_size_registered_atlas = np.array([25, 25, 25])
        self.pixel_size_data = np.array(original_pixel_size) # Z Y X
        self.merged_data = merged_data
        self.z_downsample = self.pixel_size_registered_atlas[0] / self.pixel_size_data[0]
        self.y_downsample = self.pixel_size_registered_atlas[1] / self.pixel_size_data[1]
        self.x_downsample = self.pixel_size_registered_atlas[2] / self.pixel_size_data[2]

        if atlas is not None:
            if isinstance(atlas, str):
                self.load_atlas(atlas)
            elif isinstance(atlas, np.array):
                self.atlas = atlas
            else:
                raise TypeError('Error: atlas must be a path or a numpy array.')

    @classmethod
    def from_merger(cls,
                   merger: pipapr.stitcher.tileMerger,
                   original_pixel_size: (np.array, list)):
        """
        Constructor from a tileMerger object. Typically to perform the registration to the Atlas on
        autofluorescence data.

        Parameters
        ----------
        merger: (tileMerger) tileMerger object
        original_pixel_size: (np.array, list) pixel size in µm on the original data

        Returns
        -------
        tileAtlaser object
        """

        return cls(original_pixel_size=original_pixel_size,
                   downsample=merger.downsample,
                   atlas=None,
                   merger=merger)

    @classmethod
    def from_atlas(cls,
                  atlas: (np.array, str),
                  downsample,
                  original_pixel_size: (np.array, list)):
        """
        Constructor from a previously computed Atlas. Typically to perform postprocessing using an Atlas (e.g.
        count cells per brain region).

        Parameters
        ----------
        atlas: (np.array, str) atlas data or path for loading the atlas data
        downsample: (int) downsampling used by APRSlicer to reconstruct the lower resolution pixel data used
                            for registration to the Atlas.
        original_pixel_size: (np.array, list) pixel size in µm on the original data

        Returns
        -------

        """

        return cls(original_pixel_size=original_pixel_size,
                   downsample=downsample,
                   atlas=atlas,
                   merger=None)


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
        self.atlas = np.flip(self.atlas, 1)

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

        # If merged_data is a path we ask brainreg to work on this file
        if isinstance(self.merged_data, str):
            path_merged_data = self.merged_data
        # Else it means it's an array so we have to save it first
        else:
            path_merged_data = os.path.join(output_dir, merged_data_filename)
            imsave(path_merged_data, self.merged_data)

        command = 'brainreg {} {} -v {} {} {} --orientation {}'.format('"' + path_merged_data + '"',
                                                            '"' + atlas_dir + '"',
                                                            self.pixel_size_data[0]*self.z_downsample,
                                                            self.pixel_size_data[1]*self.y_downsample,
                                                            self.pixel_size_data[2]*self.x_downsample,
                                                            orientation)
        for key, value in kwargs.items():
            command += ' --{} {}'.format(key, value)

        # Execute brainreg
        os.system(command)

        self.load_atlas(os.path.join(atlas_dir, 'registered_atlas.tiff'))

    def get_cells_id(self, cells):
        """
        Returns the Allen Brain Atlas region ID for each cell.

        Parameters
        ----------
        cells: (array) cell positions.

        Returns
        -------
        labels: (array) containing the cell region ID.
        """
        labels = self.atlas[np.floor(cells.cells[:, 0]/self.z_downsample).astype('uint64'),
                        np.floor(cells.cells[:, 1]/self.y_downsample).astype('uint64'),
                        np.floor(cells.cells[:, 2]/self.x_downsample).astype('uint64')]

        return labels

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

    def get_ontology_mapping(self, labels, n=0):
        """
        Get the mapping between area ID and name with Allen SDK.

        Parameters
        ----------
        labels: (array) array of labels to group by ID and fetch area name.
        n: (int) number of parent area to group for.

        Returns
        -------
        area_count: (dict) area names with the counts.
        """
        rspc = ReferenceSpaceCache(25, 'annotation/ccf_2017', manifest='manifest.json')
        tree = rspc.get_structure_tree(structure_graph_id=1)
        name_map = tree.get_name_map()
        ancestor_map = tree.get_ancestor_id_map()
        area_count = {}
        n_not_found = 0
        area_unknown = {}
        id_count = {}
        for l in labels:
            try:
                ids = ancestor_map[int(l)]
            except KeyError:
                n_not_found += 1
                if 'unknown' not in area_count:
                    area_count['unknown'] = 1
                else:
                    area_count['unknown'] += 1
                if int(l) not in area_unknown:
                    area_unknown[int(l)] = 1
                else:
                    area_unknown[int(l)] += 1
                continue

            if len(ids) <= 2:
                id = ids[0]
            elif len(ids) <= n:
                id = ids[-2]
            else:
                id = ids[n if n < len(ids) - 1 else n - 1]

            # Get the name and store it
            name = name_map[id]
            if name not in area_count:
                area_count[name] = 1
            else:
                area_count[name] += 1

        # Display summary
        if n == 0:
            if n_not_found > 0:
                print('\nUnknown ontology ID found for {} objects ({:0.2f}%).'.format(n_not_found,
                                                                                      n_not_found/len(labels)*100))
                print('Unknown ontology IDs and occurrences:\n')
                print(area_unknown)
            else:
                print('\nAll objects were assigned to an atlas ontology category.\n')

        return pd.DataFrame.from_dict(area_count, orient='index')

    def get_cells_number_per_region(self, cells_id):
        """
        Retuns the number of cell per region.

        Parameters
        ----------
        cells_id: (array) cells ID (typically computed by self.get_cells_id())

        Returns
        -------
        heatmap: (array) 3D array where each brain region value is the number of cells contained in this region.
        """

        # Remove 0s
        cells_id = np.delete(cells_id, cells_id==0)
        id_count = {}
        for id in cells_id:
            if id not in id_count:
                id_count[id] = 1
            else:
                id_count[id] += 1

        heatmap = np.zeros_like(self.atlas)
        for id, counts in id_count.items():
            heatmap[self.atlas==id] = counts

        return heatmap

    def get_cells_density_per_region(self, cells_id):
        """
        Retuns the cell density (number of cell per voxel) per region.

        Parameters
        ----------
        cells_id: (array) cells ID (typically computed by self.get_cells_id())

        Returns
        -------
        heatmap: (array) 3D array where each brain region value is the cell density in this region.
        """

        # Remove 0s
        cells_id = np.delete(cells_id, cells_id == 0)
        id_count = {}
        for id in cells_id:
            if id not in id_count:
                id_count[id] = 1
            else:
                id_count[id] += 1

        heatmap = np.zeros_like(self.atlas, dtype='float64')
        for id, counts in id_count.items():
            tmp = (self.atlas == id)
            heatmap[tmp] = counts/np.sum(tmp)

        return heatmap

    def get_cells_density(self, cells, kernel_size):
        """
        Retuns the cell density (local average number of cell per voxel). The local average is computed using a gaussian
        kernel.

        Parameters
        ----------
        cells: (array) cell positions
        kernel_size: (int) radius of the gaussian for local cell density estimation

        Returns
        -------

        """

        heatmap = np.zeros((self.atlas.shape)).astype(int)
        for i in range(cells.shape[0]):
            z = int(cells[i, 0]/self.z_downsample)
            y = int(cells[i, 1]/self.y_downsample)
            x = int(cells[i, 2]/self.x_downsample)
            heatmap[z, y, x] = 1

        return gaussian(heatmap, sigma=kernel_size)