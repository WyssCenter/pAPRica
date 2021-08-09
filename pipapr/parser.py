"""
Module containing classes and functions relative to Parsing.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from glob import glob
import os
import re
import numpy as np
import pipapr

class tileParser():
    """
    Class used to parse multitile data.
    """
    def __init__(self, path, frame_size, overlap, ftype=None, nrow=None, ncol=None):
        """
        Parameters
        ----------
        path: (str) path where to look for the data.
        frame_size: (int) size of each frame (camera resolution).
        overlap: (float) expected amount of overlap in %. This value should be greater than the actual overlap or the
                 algorithm will fail.
        ftype: (str) input data type in 'apr', 'tiff2D' or 'tiff3D'
        nrow: (int) number of row for parsing COLM LOCXXX data
        ncol: (int) number of col for parsing COLM LOCXXX data
        """

        self.path = path
        if ftype is None:
            self.type = self._get_type()
        else:
            self.type = ftype

        if (nrow is not None) and (ncol is not None):
            self.tiles_list = self._get_tile_list_LOC(ncol)
        else:
            self.tiles_list = self._get_tile_list()
        self.n_tiles = len(self.tiles_list)
        if self.n_tiles == 0:
            raise FileNotFoundError('Error: no tile were found.')
        self.ncol = self._get_ncol()
        self.nrow = self._get_nrow()
        self._sort_tiles()
        self.tiles_pattern, self.tile_pattern_path = self._get_tiles_pattern()
        self.neighbors, self.n_edges = self._get_neighbors_map()
        self.neighbors_tot = self._get_total_neighbors_map()
        self.path_list = self._get_path_list()
        self.overlap = int(overlap*frame_size/100)
        self.frame_size = frame_size
        self._print_info()

        # Define some folders
        base, _ = os.path.split(self.path)
        self.folder_root = base
        self.folder_max_projs = os.path.join(base, 'max_projs')

    def _print_info(self):

        print('\n**********  PARSING DATA **********')
        print('Tiles are of type {}.'.format(self.type))
        print('{} tiles were detected.'.format(self.n_tiles))
        print('{} rows and {} columns.'.format(self.nrow, self.ncol))
        print('***********************************\n')

    def _get_type(self):
        """
        Automatically determine file type based on what's inside 'path'.

        """
        folders = glob(os.path.join(self.path, '*/'))
        files_tif = glob(os.path.join(self.path, '*.tif'))
        files_apr = glob(os.path.join(self.path, '*.apr'))
        detection = (len(folders) != 0) + (len(files_tif) != 0)+(len(files_apr) != 0)

        if detection != 1:
            raise ValueError('Error: could not determine file type automatically, please pass it to the constructor.')

        if len(folders) != 0:
            return 'tiff2D'
        elif len(files_tif) != 0:
            return 'tiff3D'
        elif len(files_apr) != 0:
            return 'apr'

    def _get_tile_list(self):
        """
        Returns a list of tiles as a dictionary
        """

        if self.type == 'apr':
            # If files are apr then their names are 'row_col.apr'
            files = glob(os.path.join(self.path, '*.apr'))
        elif self.type == 'tiff3D':
            # If files are 3D tiff then their names are 'row_col.tif'
            files = glob(os.path.join(self.path, '*.tif'))
        elif self.type == 'tiff2D':
            # If files are 2D tiff then tiff sequence are in folders with name "row_col"
            # files = [f.path for f in os.scandir(self.path) if f.is_dir()]
            files = glob(os.path.join(self.path, '*/'))
        elif self.type == 'raw':
            files = glob(os.path.join(self.path, '*.raw'))
        else:
            raise TypeError('Error: file type {} not supported.'.format(self.type))

        tiles = []
        for f in files:

            pattern_search = re.search('/(\d+)_(\d+)', f)
            if pattern_search:
                row = int(pattern_search.group(1))
                col = int(pattern_search.group(2))
            else:
                raise TypeError('Couldn''t get the column/row.')

            tile = {'path': f,
                    'row': row,
                    'col': col,
                    }
            tiles.append(tile)
        return tiles

    def _get_tile_list_LOC(self, ncol):
        """
        Returns a list of tiles as a dictionary for data saved as LOC00X.
        """

        if self.type == 'apr':
            # If files are apr then their names are 'row_col.apr'
            files = glob(os.path.join(self.path, '*.apr'))
        elif self.type == 'tiff3D':
            # If files are 3D tiff then their names are 'row_col.tif'
            files = glob(os.path.join(self.path, '*.tif'))
        elif self.type == 'tiff2D':
            # If files are 2D tiff then tiff sequence are in folders with name "row_col"
            # files = [f.path for f in os.scandir(self.path) if f.is_dir()]
            files = glob(os.path.join(self.path, '*/'))
        elif self.type == 'raw':
            files = glob(os.path.join(self.path, '*.raw'))
        else:
            raise TypeError('Error: file type {} not supported.'.format(self.type))

        tiles = []
        for f in files:

            pattern_search = re.search('/LOC(\d+)', f)
            if pattern_search:
                n = int(pattern_search.group(1))
                row = n // ncol
                col = n % ncol
            else:
                raise TypeError('Couldn''t get the column/row.')

            tile = {'path': f,
                    'row': row,
                    'col': col,
                    }
            tiles.append(tile)
        return tiles

    def _sort_tiles(self):
        """
        Sort tiles so that they are arranged in columns and rows (read from left to right and top to bottom).

        """
        tiles_sorted = []
        for v in range(self.nrow):
            for h in range(self.ncol):
                for i, t in enumerate(self.tiles_list):
                    if t['col']==h and t['row']==v:
                        tiles_sorted.append(t)
                        self.tiles_list.pop(i)
                        break

        self.tiles_list = tiles_sorted

    def _get_ncol(self):
        """
        Returns the number of columns (H) to be stitched.
        """
        ncol = 0
        for tile in self.tiles_list:
            if tile['col'] > ncol:
                ncol = tile['col']
        return ncol+1

    def _get_nrow(self):
        """
        Returns the number of rows (V) to be stitched.
        """
        nrow = 0
        for tile in self.tiles_list:
            if tile['row'] > nrow:
                nrow = tile['row']
        return nrow+1

    def _get_tiles_pattern(self):
        """
        Return the tile pattern (0 = no tile, 1 = tile)

        """
        tiles_pattern = np.zeros((self.nrow, self.ncol))
        tiles_pattern_path = np.empty((self.nrow, self.ncol), dtype=object)
        for tile in self.tiles_list:
            tiles_pattern[tile['row'], tile['col']] = 1
            tiles_pattern_path[tile['row'], tile['col']] = tile['path']
        return tiles_pattern, tiles_pattern_path

    def _get_total_neighbors_map(self):
        """
        Return the total neighbors maps (with redundancy in the case of undirected graph).
        """

        # Initialize neighbors
        neighbors_tot = np.empty((self.nrow, self.ncol), dtype=object)
        cnt = 0
        for x in range(self.ncol):
            for y in range(self.nrow):
                if self.tiles_pattern[y, x] == 0:
                    pass
                # Fill up 2D list
                tmp = []
                if x < self.ncol-1 and self.tiles_pattern[y, x+1] == 1:
                    # EAST
                    tmp.append([y, x+1])
                    cnt += 1
                if y < self.nrow-1 and self.tiles_pattern[y+1, x] == 1:
                    # SOUTH
                    tmp.append([y+1, x])
                    cnt += 1
                if x > 0 and self.tiles_pattern[y, x-1] == 1:
                    # WEST
                    tmp.append([y, x-1])
                    cnt += 1
                if y > 0 and self.tiles_pattern[y-1, x] == 1:
                    # NORTH
                    tmp.append([y-1, x])
                    cnt += 1
                neighbors_tot[y, x] = tmp
        return neighbors_tot

    def _get_neighbors_map(self):
        """
        Returns the non-redundant neighbors map: neighbors[row, col] gives a list of neighbors and the total
        number of pair-wise neighbors. Only SOUTH and EAST are returned to avoid the redundancy.
        """

        # Initialize neighbors
        neighbors = np.empty((self.nrow, self.ncol), dtype=object)
        cnt = 0
        for x in range(self.ncol):
            for y in range(self.nrow):
                if self.tiles_pattern[y, x] == 0:
                    pass
                # Fill up 2D list
                tmp = []
                if x < self.ncol-1 and self.tiles_pattern[y, x+1] == 1:
                    # EAST
                    tmp.append([y, x+1])
                    cnt += 1
                if y < self.nrow-1 and self.tiles_pattern[y+1, x] == 1:
                    # SOUTH
                    tmp.append([y+1, x])
                    cnt += 1
                neighbors[y, x] = tmp

        return neighbors, cnt

    def _get_path_list(self):
        """
        Returns a list containing the path to each tile.
        """
        path_list = []
        for tile in self.tiles_list:
            path_list.append(tile['path'])
        return path_list

    def __getitem__(self, item):
        """
        Return tiles, add neighbors information before returning.
        """

        t = self.tiles_list[item]
        path = t['path']
        col = t['col']
        row = t['row']
        neighbors = self.neighbors[row, col]
        neighbors_tot = self.neighbors_tot[row, col]

        neighbors_path = []
        for r, c in neighbors:
            if self.tiles_pattern[r, c]:
                neighbors_path.append(self.tile_pattern_path[r, c])

        return pipapr.loader.tileLoader(path=path,
                                          row=row,
                                          col=col,
                                          ftype=self.type,
                                          neighbors=neighbors,
                                          neighbors_tot=neighbors_tot,
                                          neighbors_path=neighbors_path,
                                          overlap=self.overlap,
                                          frame_size=self.frame_size,
                                          folder_root=self.folder_root)

    def __len__(self):
        """
        Returns the number of tiles.
        """
        return self.n_tiles


class randomParser():
    """
    Class used to parse several independent tiles (not multitile).
    """

    def __init__(self, path, frame_size, ftype):
        self.path = path
        self.type = ftype
        self.tiles_list = self._get_tile_list()
        self.n_tiles = len(self.tiles_list)
        self.ncol = None
        self.nrow = None
        self.tiles_pattern, self.tile_pattern_path = None, None
        self.neighbors, self.n_edges = None, None
        self.path_list = self._get_path_list()
        self.overlap = None
        self.frame_size = frame_size
        self._print_info()

        # Define some folders
        base, _ = os.path.split(self.path)
        self.folder_root = base
        self.folder_max_projs = None

    def _print_info(self):

        print('\n**********  PARSING DATA **********')
        print('Tiles are of type {}.'.format(self.type))
        print('{} tiles were detected.'.format(self.n_tiles))
        print('***********************************\n')

    def _get_tile_list(self):
        """
        Returns a list of tiles as a dictionary
        """

        if self.type == 'apr':
            # If files are apr then their names are 'row_col.apr'
            files = glob(os.path.join(self.path, '*.apr'))
        elif self.type == 'tiff3D':
            # If files are 3D tiff then their names are 'row_col.tif'
            files = glob(os.path.join(self.path, '*.tif'))
        elif self.type == 'tiff2D':
            # If files are 2D tiff then tiff sequence are in folders with name "row_col"
            # files = [f.path for f in os.scandir(self.path) if f.is_dir()]
            files = glob(os.path.join(self.path, '*/'))
        elif self.type == 'raw':
            files = glob(os.path.join(self.path, '*.raw'))
        else:
            raise TypeError('Error: file type {} not supported.'.format(self.type))

        tiles = []
        for f in files:
            tile = {'path': f,
                    'row': None,
                    'col': None,
                    }
            tiles.append(tile)
        return tiles

    def _get_path_list(self):
        """
        Returns a list containing the path to each tile.
        """
        path_list = []
        for tile in self.tiles_list:
            path_list.append(tile['path'])
        return path_list

    def __getitem__(self, item):
        """
        Return tiles, add neighbors information before returning.
        """

        t = self.tiles_list[item]
        path = t['path']
        col = t['col']
        row = t['row']
        neighbors = None

        neighbors_path = None

        return pipapr.loader.tileLoader(path=path,
                                          row=row,
                                          col=col,
                                          ftype=self.type,
                                          neighbors=neighbors,
                                          neighbors_path=neighbors_path,
                                          overlap=self.overlap,
                                          frame_size=self.frame_size,
                                          folder_root=self.folder_root)

    def __len__(self):
        """
        Returns the number of tiles.
        """
        return self.n_tiles