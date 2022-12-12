"""
Submodule containing classes and functions relative to **atlasing**.

This submodule is essentially a wrapper to Brainreg (https://github.com/brainglobe/brainreg) for atlasing and
Allen Brain Atlas for ontology analysis. It contains many convenience method for manipulating data
(per region, per superpixel, etc.).

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from skimage.filters import gaussian
from skimage.io import imread, imsave
from tqdm import tqdm

import paprica


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
        original_pixel_size: array_like
            pixel size in µm on the original data
        downsample: int
            downsampling used by APRSlicer to reconstruct the lower resolution pixel data used
            for registration to the Atlas.
        atlas: ndarray, string
            atlas data or path for loading the atlas data
        merger: tileMerger
            tileMerger object

        Returns
        -------
        None
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
                   merger: paprica.stitcher.tileMerger,
                   original_pixel_size: (np.array, list)):
        """
        Constructor from a tileMerger object. Typically to perform the registration to the Atlas on
        autofluorescence data.

        Parameters
        ----------
        merger: tileMerger
            tileMerger object
        original_pixel_size: array_like
            pixel size in µm on the original data

        Returns
        -------
        tileAtlaser object
        """

        return cls(original_pixel_size=original_pixel_size,
                   downsample=merger.downsample,
                   atlas=None,
                   merged_data=merger.merged_data)

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
        atlas: ndarray, string
            atlas data or path for loading the atlas data
        downsample: int
            downsampling used by APRSlicer to reconstruct the lower resolution pixel data used
            for registration to the Atlas.
        original_pixel_size: array_like
            pixel size in µm on the original data

        Returns
        -------
        tileAtlaser object
        """

        return cls(original_pixel_size=original_pixel_size,
                   downsample=downsample,
                   atlas=atlas,
                   merged_data=None)

    def load_atlas(self, path):
        """
        Function to load a previously computed atlas.

        Parameters
        ----------
        path: string
            path to the registered atlas file.

        Returns
        -------
        None
        """

        self.atlas = imread(path)

    def register_to_atlas(self,
                          output_dir='./',
                          orientation='spr',
                          merged_data_filename='merged_data.tif',
                          debug=False,
                          params=None):
        """
        Function to compute the registration to the Atlas. It is just a wrapper to call brainreg.

        Parameters
        ----------
        output_dir: string
            output directory to save atlas
        orientation: string
            orientation of the input data with respect to the origin in Z,Y,X order. E.g. 'spr' means
            superior (so going from origin to z = zlim we go from superior to inferior), posterior
            (so going from origin to y = ylim we go from posterior to anterior part) and right (so going
            from origin to x = xlim we go from right to left part)
        merged_data_filename: string
            named of the merged array (Brainreg reads data from files so we need to save
            the merged volume beforehand.
        debug: bool
            add debug option for brainreg which will save intermediate steps.
        params: dict
            dictionary with keys as brainreg options and values as parameters (see here:
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

        command = 'brainreg {} {} -v {} {} {} --orientation {} --save-original-orientation'.format('"' + path_merged_data + '"',
                                                            '"' + atlas_dir + '"',
                                                            self.pixel_size_data[0]*self.downsample,
                                                            self.pixel_size_data[1]*self.downsample,
                                                            self.pixel_size_data[2]*self.downsample,
                                                            orientation)

        if params is not None:
            for key, value in params.items():
                command += ' --{} {}'.format(key, value)

        if debug:
            command += ' --debug'

        # Execute brainreg
        os.system(command)

        self.load_atlas(os.path.join(atlas_dir, 'registered_atlas.tiff'))

    def get_cells_id(self, cells):
        """
        Returns the Allen Brain Atlas region ID for each cell.

        Parameters
        ----------
        cells: ndarray
            cell positions

        Returns
        -------
        labels: ndarray
            containing the cell region ID.
        """
        cells_id = self.atlas[np.floor(cells[:, 0]/self.z_downsample).astype('uint64'),
                        np.floor(cells[:, 1]/self.y_downsample).astype('uint64'),
                        np.floor(cells[:, 2]/self.x_downsample).astype('uint64')]

        return cells_id

    def get_loc_id(self, x, y, z):
        """
        Return the ID (brain region) of a given position (typically to retrieve cell position in the brain).

        Parameters
        ----------
        x: int
            x position
        y: int
            y position
        z: int
            z position

        Returns
        -------
        _: list
            ID at the queried position.
        """
        return self.atlas[int(z / self.z_downsample), int(y / self.y_downsample), int(x / self.x_downsample)]

    def get_ontology_mapping(self, labels, n=0):
        """
        Get the mapping between area ID and name with Allen SDK.

        Parameters
        ----------
        labels: ndarray
            array of labels to group by ID and fetch area name.
        n: int
            number of parent area to group for.

        Returns
        -------
        area_count: dict
            area names with the counts.
        """
        rspc = ReferenceSpaceCache(25, 'annotation/ccf_2017', manifest='manifest.json')
        tree = rspc.get_structure_tree(structure_graph_id=1)
        name_map = tree.get_name_map()
        ancestor_map = tree.get_ancestor_id_map()
        area_count = {}
        n_not_found = 0
        area_unknown = {}
        for l in labels:
            # Fetch the ancestor map of the label.
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

            # At the bottom of the tree each regions has 10 ancestors, with 'root' being the higher
            n_ancestors = len(ids)
            n_start = 10 - n_ancestors
            if n <= n_start:
                id = ids[0]
            elif n > n_start:
                if n_ancestors > n - n_start + 1:
                    id = ids[n - n_start]
                else:
                    id = ids[-2]

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
                                                                                      100 * n_not_found / len(labels)))
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
        cells_id: ndarray
            cells ID (typically computed by self.get_cells_id())

        Returns
        -------
        heatmap: ndarray
            3D array where each brain region value is the number of cells contained in this region.
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
        cells_id: ndarray
            cells ID (typically computed by self.get_cells_id())

        Returns
        -------
        heatmap: ndarray
            3D array where each brain region value is the cell density in this region.
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

    def get_cells_density(self, cells, kernel_size, progress_bar=True):
        """
        Retuns the cell density (local average number of cell per voxel). The local average is computed using a gaussian
        kernel.

        Parameters
        ----------
        cells: ndarray
            cell positions
        kernel_size: int
            radius of the gaussian for local cell density estimation

        Returns
        -------
        _: ndarray
            estimated cell density
        """

        heatmap = np.zeros((self.atlas.shape)).astype(int)
        for i in tqdm(range(cells.shape[0]), desc='Building density map..', disable=not progress_bar):
            z = int(cells[i, 0]/self.z_downsample)
            y = int(cells[i, 1]/self.y_downsample)
            x = int(cells[i, 2]/self.x_downsample)
            heatmap[z, y, x] = 1

        return gaussian(heatmap, sigma=kernel_size)

    def get_cell_number_by_acronym(self, acronym_list, cells_ids):
        """
        Get the total number of segmented cell in different regions referenced by their acronyms.

        Parameters
        ----------
        acronym_list: list
            list of acronyms (ABA)
        cells_ids: ndarray
            cells ID (typically computed by self.get_cells_id())

        Returns
        -------
        cell_number: arraylike
            list containing the total number of cells for each asked region.
        """

        if isinstance(acronym_list, str):
            acronym_list = [acronym_list]

        rspc = ReferenceSpaceCache(25, 'annotation/ccf_2017', manifest='manifest.json')
        tree = rspc.get_structure_tree(structure_graph_id=1)

        structures = tree.get_structures_by_acronym(acronym_list)

        cell_number = []
        for structure in structures:
            ids = tree.descendant_ids([structure['id']])
            cell_number.append(np.sum(np.isin(element=cells_ids, test_elements=ids)))

        return cell_number

    def get_area_mask_by_acronym(self, acronym_list):
        """
        Return the mask corresponding to brain regions given in `acronym_list`, brain regions referred by their
        Allen brain acronym.

        Parameters
        ----------
        acronym_list: list
            list of Allen Brain region acronyms to count the mapped cells.

        Returns
        -------
        mask: ndarray
            mask containing `1` for the region in `acronym_list` ans `0` elsewhere.
        """


        rspc = ReferenceSpaceCache(25, 'annotation/ccf_2017', manifest='manifest.json')
        tree = rspc.get_structure_tree(structure_graph_id=1)

        structures = tree.get_structures_by_acronym(acronym_list)

        ids = []
        for structure in structures:
            ids.extend(tree.descendant_ids([structure['id']]))

        ids = np.array(ids)
        mask = np.isin(element=self.atlas, test_elements=ids)

        return mask