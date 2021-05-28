"""
Module containing classes and functions relative to Parsing.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from glob import glob
import os
import re
import numpy as np

class tileParser():
    """
    Class used to parse the data.
    """
    def __init__(self, path, frame_size, overlap, ftype=None):
        self.path = path
        if ftype is None:
            self.type = self._get_type()
        else:
            self.type = ftype
        self.tiles_list = self._get_tile_list()
        self.n_tiles = len(self.tiles_list)
        self.ncol = self._get_ncol()
        self.nrow = self._get_nrow()
        self._sort_tiles()
        self.tiles_pattern, self.tile_pattern_path = self._get_tiles_pattern()
        self.neighbors, self.n_edges = self._get_neighbors_map()
        self.path_list = self._get_path_list()
        self.overlap = overlap
        self.frame_size = frame_size
        self._print_info()

    def _print_info(self):

        print('\n**********  PARSING DATA **********')
        print('Tiles are of type {}.'.format(self.type))
        print('{} tiles were detected.'.format(self.n_tiles))
        print('{} rows and {} columns.'.format(self.nrow, self.ncol))
        print('***********************************')

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

        Note: this function is not used anymore.
        """
        # Initialize neighbors
        neighbors = [None] * self.ncol
        for x in range(self.ncol):
            # Create new dimension
            neighbors[x] = [None] * self.nrow
            for y in range(self.nrow):
                # Fill up 2D list
                tmp = []
                if x > 0:
                    # NORTH
                    tmp.append([x-1, y])
                if x < self.ncol-1:
                    # SOUTH
                    tmp.append([x+1, y])
                if y > 0:
                    # WEST
                    tmp.append([x, y-1])
                if y < self.nrow-1:
                    # EAST
                    tmp.append([x, y+1])
                neighbors[x][y] = tmp
        return neighbors

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
                    # SOUTH
                    tmp.append([y, x+1])
                    cnt += 1
                if y < self.nrow-1 and self.tiles_pattern[y+1, x] == 1:
                    # EAST
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

        e = self.tiles_list[item]
        e['neighbors'] = self.neighbors[e['row'], e['col']]
        neighbors_path = []
        for row, col in e['neighbors']:
            if self.tiles_pattern[row, col]:
                neighbors_path.append(self.tile_pattern_path[row, col])
        e['neighbors_path'] = neighbors_path
        e['type'] = self.type
        e['overlap'] = self.overlap
        e['frame_size'] = self.frame_size
        return e

    def __len__(self):
        """
        Returns the number of tiles.
        """
        return self.n_tiles