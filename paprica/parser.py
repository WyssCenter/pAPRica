"""
Submodule containing classes and functions relative to Parsing.

The general idea of this submodule is to parse the data to be processed later on. This submodule was developed for
our particular folder layout and was particularly adapted for COLM, mesoSPIM and ClearScope.

Note that each channel is parsed separately so as to give maximum flexibility for stitching and visualization.

There are two general way of parsing the data:
- **multitile** parsing (tileParser class), where each tile has a given position on a 2D grid and can therefore be stitched
- **independant** parsing (baseParser class), where each tile is independent

We also provide a few classes to parse data from given microscopes:
- COLM
- MesoSpim
- ClearScope

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import copy
import inspect
import os
import re
from glob import glob
import warnings

import pyapr
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

import paprica

def get_microscope_list():
    """
    This function builds up a dict containing microscopes supported by the pipeline (refer to the documentation to add yours).
    By default, this list should be equivalent to:

    >>> microscope_list = {'colm': colmParser,
    >>>                   'clearscope': clearscopeParser,
    >>>                   'default': tileParser
    >>>                    }

    The list is built by looking at the class that inherits from the base parser class and that are not the base parser class.

    Returns
    -------
    microscope_list: dict
        dictionnary containing all supported microscopes.
    """
    microscope_list = {}
    for name, obj in inspect.getmembers(paprica.parser):
        if inspect.isclass(obj):
            if issubclass(obj, paprica.parser.baseParser) and (obj != paprica.parser.baseParser):
                microscope_list[name] = obj
    return microscope_list

def autoParser(path, **kwargs):
    """
    This function allows to parse a data-set automatically by guessing the folder architecture.
    This function uses the microscope_list that is automatically infered from this submodule.
    For each microscope it will call the __is_valid_acquisition() method and return the according parser object.

    Parameters
    ----------
    path: str
        path containing the data

    Returns
    -------
    tileParser object
    """
    for acq_type, parser in get_microscope_list().items():
        if parser._is_valid_acquisition(path):
            return parser(path, **kwargs)

def get_number_of_channels(path):
    """
    This functions returns the number of channels acquired in an acquisition located in ´path´ folder.
    Parameters
    ----------
    path: str
        path to the folder to check the number of channels

    Returns
    -------
    Number of acquired channels
    """
    for acq_type, parser in get_microscope_list().items():
        if parser._is_valid_acquisition(path):
            return parser._get_n_channels(path)


class baseParser():
    """
    Class used to parse several independent tiles (not multitile).

    """
    def __init__(self, path, frame_size, ftype, verbose=True):
        """
        Constructor of the baseParser object.

        Parameters
        ----------
        path: string
            path where to look for the data.
        frame_size: int
            size of each frame (camera resolution).
        ftype: string
            input data type in 'apr', 'tiff2D' or 'tiff3D'

        """
        self.path = path
        self.frame_size = frame_size
        self.type = ftype
        self.channel = None
        self.tiles_list = self._get_tile_list()
        self.n_tiles = len(self.tiles_list)
        self.ncol = None
        self.nrow = None
        self.tiles_pattern, self.tile_pattern_path = None, None
        self.neighbors, self.n_edges = None, None
        self.neighbors_tot = None
        self.neighbors_path = None
        self.path_list = self._get_path_list()
        if verbose:
            self._print_info()

        # Define some folders
        base, _ = os.path.split(self.path)
        self.folder_root = base
        self.folder_max_projs = None

    def check_files_integrity(self):
        """
        Check that all tiles are readable and not corrupted.

        Returns
        -------
        None
        """
        cnt = 0
        for tile in self:
            lazy = 1
            try:
                tile.lazy_load_tile()
            except:
                lazy = 0
                print('Lazy load failed on ({}, {})\n Trying normal loading...'.format(tile.row, tile.col))

            if lazy == 0:
                try:
                    tile.load_tile()
                except:
                    cnt += 1
                    print('Problem detected with tile ({}, {})'.format(tile.row, tile.col))

        if cnt == 0:
            print('All tiles are readable.')

    def compute_average_CR(self, progress_bar=True):
        """
        Compute the average Computational Ratio (CR). Note: data must be of type APR.

        Returns
        -------
        cr: float
            average CR for the dataset
        """

        if self.type != 'apr':
            warnings.warn('Data-set should be of type APR to compute CR, returning 1.')
            return 1

        n_parts = []
        n_pixels = []
        try:
            for tile in tqdm(self, desc='Computing CR', disable=not progress_bar):
                tile.lazy_load_tile()
                n_parts.append(tile.lazy_data.parts.dataset_size())
                n_pixels.append(np.prod(tile.lazy_data.shape))

        except: #Lazy loading not available
            for tile in tqdm(self, desc='Computing CR', disable=not progress_bar):
                tile.load_tile()
                n_parts.append(len(tile.parts))
                n_pixels.append(np.prod(tile.apr.shape()))

        return np.sum(n_pixels)/np.sum(n_parts)

    def make_lazy_loadable(self):
        """
        Loads all parsed APR tiles and compute and save the tree parts so that the data-set becomes lazy loadable.

        Returns
        -------
        None
        """

        if self.type != 'apr':
            raise TypeError('Only APR data can be made lazy loadable.')

        for tile in tqdm(self, desc='Making tiles lazy loadable.'):
            tile.load_tile()
            # Compute tree parts
            tree_parts = pyapr.tree.fill_tree_mean(tile.apr, tile.parts)
            # Save tree parts
            pyapr.io.write_particles(tile.path, tree_parts, parts_name='particles', tree=True, append=True)

    def _print_info(self):
        """
        Display parsing summary in the terminal.

        """
        print('\n**********  PARSING DATA **********')
        print('Tiles are of type {}.'.format(self.type))
        print('{} tiles were detected.'.format(self.n_tiles))
        print('***********************************\n')

    def _correct_offset(self):
        """
        If the row or column do not start at 0, then we subtract the min_row and min_col so that it starts at 0.

        """

        for i, tile in enumerate(self.tiles_list):
            if i == 0:
                row_min = tile['row']
                col_min = tile['col']

            row_min = min(row_min, tile['row'])
            col_min = min(col_min, tile['col'])

        if (row_min > 0) or (col_min > 0):
            for tile in self.tiles_list:
                tile['row'] -= row_min
                tile['col'] -= col_min

    def _get_tile_list(self):
        """
        Returns a list of tiles as a dictionary

        """

        files = self._get_tiles_path()
        return self._get_tiles_from_path(files)

    def _get_tiles_path(self):
        """
        Returns a list containing file paths (for tiff3D and APR) or folder paths (for tiff2).

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

        return files

    def _get_tiles_from_path(self, files):
        """
        Create a list of dictionnary for each tile containing it's path and coordinate on the grid.
        Coordinates are set to None for the baseParser which only parse independant tiles.

        """

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

    def __getitem__(self, item):
        """
        Return tiles, add neighbors information before returning.

        """
        t = self.tiles_list[item]
        path = t['path']
        col = t['col']
        row = t['row']

        return paprica.loader.tileLoader(path=path,
                                         row=row,
                                         col=col,
                                         ftype=self.type,
                                         neighbors=self.neighbors,
                                         neighbors_tot=self.neighbors_tot,
                                         neighbors_path=self.neighbors_path,
                                         frame_size=self.frame_size,
                                         folder_root=self.folder_root,
                                         channel=self.channel)

    def __iter__(self):
        """
        Method called under the hood when iterating on tiles. Because it is a generator, the memory is released at
        each iteration.

        Returns
        -------
        Generator containing the tileLoader object.
        """
        for i in range(self.n_tiles):
                t = self.tiles_list[i]
                path = t['path']
                col = t['col']
                row = t['row']

                yield paprica.loader.tileLoader(path=path,
                                                row=row,
                                                col=col,
                                                ftype=self.type,
                                                neighbors=self.neighbors,
                                                neighbors_tot=self.neighbors_tot,
                                                neighbors_path=self.neighbors_path,
                                                frame_size=self.frame_size,
                                                folder_root=self.folder_root,
                                                channel=self.channel)

    def __len__(self):
        """
        Returns the number of tiles.

        """
        return self.n_tiles


class tileParser(baseParser):
    """
    Class used to parse multi-tile data where each tile position in space matters. Tile parsed this way are usually
    stitched later on.

    """
    def __init__(self, path, frame_size=2048, ftype=None, verbose=True):
        """
        Constructor of the tileParser object.

        Parameters
        ----------
        path: string
            path where to look for the data.
        frame_size: int
            size of each frame (camera resolution).
        ftype: string
            input data type in 'apr', 'tiff2D' or 'tiff3D'

        """

        self.path = path
        self.frame_size = frame_size
        self.channel = None
        if ftype is None:
            self.type = self._get_type()
        else:
            self.type = ftype

        self.tiles_list = self._get_tile_list()
        self.n_tiles = len(self.tiles_list)
        if self.n_tiles == 0:
            raise FileNotFoundError('Error: no tile were found.')

        self._correct_offset()
        self.ncol = self._get_ncol()
        self.nrow = self._get_nrow()
        self._sort_tiles()
        self.tiles_pattern, self.tile_pattern_path = self._get_tiles_pattern()
        self.neighbors, self.n_edges = self._get_neighbors_map()
        self.neighbors_tot = self._get_total_neighbors_map()
        self.path_list = self._get_path_list()
        if verbose:
            self._print_info()

        # Define some folders
        base, _ = os.path.split(self.path)
        self.folder_root = base
        self.folder_max_projs = os.path.join(base, 'max_projs')

    def _print_info(self):
        """
        Display parsing summary in the terminal.

        """
        print('\n**********  PARSING DATA **********')
        print('{}'.format(self.path))
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

    def _get_tiles_from_path(self, files):
        """
        Create a list of dictionnary for each tile containing it's path and coordinate on the grid.

        """
        tiles = []
        for f in files:
            _, fname = os.path.split(f)
            pattern_search = re.search(r'(\d+)_(\d+).', fname)
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

    def __getitem__(self, item):
        """
        If item is an int, then returns the corresponding tileLoader object.

        If item is a tuple, then returns the corresponding (row, col) tileLoader object.

        If item is a slice, then it creates a generator so the tileLoader object is garbage collected at each iteration.

        """

        if isinstance(item, tuple):
            if (item[0] > self.nrow) or (item[0] < 0):
                raise ValueError('Error: tile at requested coordinates does not exists.')
            if (item[1] > self.ncol) or (item[1] < 0):
                raise ValueError('Error: tile at requested coordinates does not exists.')
            t = self.tiles_list[item[0]*self.ncol + item[1]]
            path = t['path']
            col = t['col']
            row = t['row']
            neighbors = self.neighbors[row, col]
            neighbors_tot = self.neighbors_tot[row, col]

            neighbors_path = []
            for r, c in neighbors:
                if self.tiles_pattern[r, c]:
                    neighbors_path.append(self.tile_pattern_path[r, c])

            return paprica.loader.tileLoader(path=path,
                                             row=row,
                                             col=col,
                                             ftype=self.type,
                                             neighbors=neighbors,
                                             neighbors_tot=neighbors_tot,
                                             neighbors_path=neighbors_path,
                                             frame_size=self.frame_size,
                                             folder_root=self.folder_root,
                                             channel=self.channel)

        elif isinstance(item, int):
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

            return paprica.loader.tileLoader(path=path,
                                             row=row,
                                             col=col,
                                             ftype=self.type,
                                             neighbors=neighbors,
                                             neighbors_tot=neighbors_tot,
                                             neighbors_path=neighbors_path,
                                             frame_size=self.frame_size,
                                             folder_root=self.folder_root,
                                             channel=self.channel)

        elif isinstance(item, slice):
            sliced_parser = copy.copy(self)
            sliced_parser.tiles_list = sliced_parser.tiles_list[item]
            sliced_parser.n_tiles = len(sliced_parser.tiles_list)
            return iter(sliced_parser)

    def __iter__(self):
        """
        Method called under the hood when iterating on tiles. Because it is a generator, the memory is released at
        each iteration.

        Returns
        -------
        Generator containing the tileLoader object.
        """
        for i in range(self.n_tiles):
                t = self.tiles_list[i]
                path = t['path']
                col = t['col']
                row = t['row']
                neighbors = self.neighbors[row, col]
                neighbors_tot = self.neighbors_tot[row, col]

                neighbors_path = []
                for r, c in neighbors:
                    if self.tiles_pattern[r, c]:
                        neighbors_path.append(self.tile_pattern_path[r, c])

                yield paprica.loader.tileLoader(path=path,
                                                row=row,
                                                col=col,
                                                ftype=self.type,
                                                neighbors=neighbors,
                                                neighbors_tot=neighbors_tot,
                                                neighbors_path=neighbors_path,
                                                frame_size=self.frame_size,
                                                folder_root=self.folder_root,
                                                channel=self.channel)

    @staticmethod
    def _is_valid_acquisition(path):
        for f in os.listdir(path):
            if f.endswith('.apr'):
                return True
        else:
            return False

    @staticmethod
    def _get_n_channels(path):
        """
        Returns the number of channel of the CLEARSCOPE acquisition. This is based on a regex that checks
        the highest n in the path: some_path/X_Y___nx/
        """
        folders = glob(os.path.basename(path))
        return np.max([int(re.findall(r'ch(\d)', x)[0])+1 for x in folders])


class colmParser(tileParser):
    """
    Class used to parse multi-tile colm data where each tile position in space matters. Tile parsed this way are usually
    stitched later on.

    """
    def __init__(self, path, channel=0, verbose=True):
        """
        Constructor of the tileParser object for COLM acquisition.

        Parameters
        ----------
        path: string
            path where to look for the data. More specifically it should be the folder that contains the acquisition.
        channel: int
            fluorescence channel for parsing COLM LOCXXX data
        verbose: bool
            Control verbosity of the parsing. If True, the parser will print acquisition info in the terminal.

        """

        u = np.loadtxt(os.path.join(path, 'Scanned Cells.txt'), delimiter=',')
        self.ncol = u.shape[1]
        self.nrow = u.shape[0]
        path = os.path.join(path, 'VW0')
        super().__init__(path, frame_size=2048, ftype='colm', verbose=verbose)

        self.channel = channel

    def _get_tiles_path(self):
        """
        Returns a list containing COLM folders which contains individual tiff.

        """
        return glob(os.path.join(self.path, '*/'))

    def _get_tiles_from_path(self, files):
        """
        Returns a list of tiles as a dictionary for data saved as LOC00X.

        """
        tiles = []
        for f in files:
            pattern_search = re.findall(r'[\\\/]LOC(\d+)', f)
            if pattern_search != []:
                n = int(pattern_search[-1])
                row = n // self.ncol
                col = n % self.ncol
            else:
                raise TypeError('Couldn''t get the column/row.')

            tile = {'path': f,
                    'row': row,
                    'col': col,
                    }
            tiles.append(tile)
        return tiles

    def get_overlap(self):
        """
        Extract overlap from COLM Experiment.ini file.

        """

        with open(os.path.join(self.path[:-4], 'Experiment.ini')) as f:
            lines = f.readlines()

        acq_param = {}
        for l in lines:
            pattern_matched = re.match(r'(.*) = "(.*)"$', l)
            if pattern_matched is not None:
                if self._is_number(pattern_matched.group(2)):
                    acq_param[pattern_matched.group(1)] = float(pattern_matched.group(2))
                elif pattern_matched.group(2) == 'True':
                    acq_param[pattern_matched.group(1)] = True
                elif pattern_matched.group(2) == 'False':
                    acq_param[pattern_matched.group(1)] = False
                else:
                    acq_param[pattern_matched.group(1)] = pattern_matched.group(2)

        return acq_param['Actual Vertical Overlap (%)'], acq_param['Actual Horizontal Overlap (%)']

    @staticmethod
    def _is_valid_acquisition(path):
        """
        This function returns True if path folder is a COLM acquisition.
        Parameters
        ----------
        path: str
            path to check if acquisition is COLM

        Returns
        -------
        bool
        """
        return os.path.exists(os.path.join(path, 'Experiment.ini'))

    @staticmethod
    def _get_n_channels(path):
        """
        Returns the number of channel of the COLM acquisition.
        """
        files = sorted(glob(os.path.join(path, 'VW0', 'LOC000', '*CHN*.tif')))
        return int(re.findall(r'CHN(\d+)', files[-1])[0])+1

    @staticmethod
    def _is_number(s):
        """
        Check if a string contain any king of number (isnumeric() method doesn't work for decimal numbers).
        Returns
        -------
        True if s is a number, False otherwise
        """
        try:
            float(s)
            return True
        except ValueError:
            return False


class clearscopeParser(tileParser):
    """
    Class used to parse multi-tile colm data where each tile position in space matters. Tile parsed this way are usually
    stitched later on.

    """
    def __init__(self, path, channel=0, verbose=True):
        """
        Constructor of the tileParser object for COLM acquisition.

        Parameters
        ----------
        path: string
            path where to look for the data.
        channel: int
            fluorescence channel for parsing CLEARSCOPE data
        verbose: bool
            Control verbosity of the parsing. If True, the parser will print acquisition info in the terminal.

        """

        self.path = os.path.join(path, '0001')
        self.channel = channel
        self.folder_settings, self.name_acq = os.path.split(path)
        self._parse_settings()
        self.frame_size = 2048
        self.type = 'clearscope'
        self.tiles_list = self._get_tile_list()
        self.n_tiles = len(self.tiles_list)
        if self.n_tiles == 0:
            raise FileNotFoundError('Error: no tile were found.')

        self._sort_tiles()
        self.tiles_pattern, self.tile_pattern_path = self._get_tiles_pattern()
        self.neighbors, self.n_edges = self._get_neighbors_map()
        self.neighbors_tot = self._get_total_neighbors_map()
        self.path_list = self._get_path_list()
        if verbose:
            self._print_info()

        # Define some folders
        base, _ = os.path.split(self.path)
        self.folder_root = base
        self.folder_max_projs = os.path.join(base, 'max_projs')

    def _parse_settings(self):

        path = os.path.join(self.folder_settings, '{}_AcquireSettings.txt'.format(self.name_acq))
        print(path)
        print('Settings found: {}'.format(path))


        with open(path) as f:
            lines = f.readlines()

        self.acq_param = {}
        for l in lines:
            pattern_matched = re.match(r'^(\w*) = (.*)$', l)
            if pattern_matched is not None:
                if pattern_matched.group(2).isnumeric():
                    self.acq_param[pattern_matched.group(1)] = float(pattern_matched.group(2))
                elif pattern_matched.group(2) == 'True':
                    self.acq_param[pattern_matched.group(1)] = True
                elif pattern_matched.group(2) == 'False':
                    self.acq_param[pattern_matched.group(1)] = False
                else:
                    self.acq_param[pattern_matched.group(1)] = pattern_matched.group(2)

        self.nrow = int(self.acq_param['ScanGridY'])
        self.ncol = int(self.acq_param['ScanGridX'])
        self.n_tiles = self.nrow*self.ncol
        self.n_planes = int(self.acq_param['StackDepths'])

    def _get_tiles_path(self):
        """
        Returns a list containing ClearScope folders which contains individual tiff.

        """
        return glob(os.path.join(self.path, '000000_*_{}c/'.format(self.channel)))

    def _get_tiles_from_path(self, files):
        """
        Returns a list of tiles as a dictionary for ClearScope data.

        """
        tiles = []
        for f in files:

            pattern_search = re.findall(r'\d{6}_(\d{6})___\dc', f)
            if pattern_search != []:
                n = int(pattern_search[0])
                row, col = self._get_row_col(n)
            else:
                raise TypeError('Couldn''t get the column/row.')

            tile = {'path': f,
                    'row': row,
                    'col': col,
                    }
            tiles.append(tile)
        return tiles

    def _get_row_col(self, n):
        """
        Get ClearScope tile row and col position given the tile number.

        Parameters
        ----------
        n: int
            ClearScope tile number

        Returns
        -------
        row: int
            row number
        col: int
            col number
        """

        col = np.absolute(np.mod(n - self.ncol - 1, 2 * self.ncol) - self.ncol + 0.5) + 0.5
        row = np.ceil(n / self.ncol)

        col = int(col-1)
        row = int(row-1)

        return row, col

    def _find_missing_frames(self):

        folders = sorted(self._get_tiles_path())

        missing_frames = np.zeros(len(folders))
        for i, folder in enumerate(folders):
            files = sorted(glob(os.path.join(folder, '*.tif')))

            n = []
            for file in files:
                pattern_search = re.findall(r'\d{6}_\d{6}___(\d{6})_\dc.tif', file)
                n.append(int(pattern_search[0]))

            n = np.array(n)
            dn = np.diff(n)
            inds = np.where(dn > 1)[0]

            if len(inds)>0:
                print('\nTile {}'.format(folder))
            for ind in inds:
                print('Missing {} frames at index {}'.format(dn[ind] - 1, ind+1))

            missing_frames[i] = self.n_planes - len(files)

        return missing_frames

    def interpolate_missing_frames(self):
        """
        Interpolate missing frames and save them.

        Returns
        -------
        None
        """

        folders = sorted(self._get_tiles_path())

        for i, folder in enumerate(folders):

            # First we build a list of all filenames
            files = sorted(glob(os.path.join(folder, '*.tif')))
            list_of_filenames = []
            for file in files:
                _, filename = os.path.split(file)
                list_of_filenames.append(filename)

            # Then we check if some are missing
            patterns = re.findall(r'(\d{6})_(\d{6})___\d{6}_(\d)c.tif', files[0])
            patterns = patterns[0]
            a1 = patterns[0]
            a2 = patterns[1]
            a3 = patterns[2]
            missing_ind = []
            for ii in range(self.n_planes):
                expected_filename = '{}_{}___{:06d}_{}c.tif'.format(a1, a2, ii, a3)
                if expected_filename not in list_of_filenames:
                    print('Missing frames: {}'.format(expected_filename))
                    missing_ind.append(ii)

            # Check if there are contiguous numbers:
            if missing_ind != []:
                cnt = 0
                cnt2 = 1
                list_of_ind = [[missing_ind[0]]]
                while cnt < len(missing_ind)-1:
                    if missing_ind[cnt+1] - missing_ind[cnt] == 1:
                        if len(list_of_ind) < cnt2:
                            list_of_ind.append([])
                        list_of_ind[cnt2-1].append(missing_ind[cnt+1])
                    else:
                        cnt2 += 1
                        list_of_ind.append([missing_ind[cnt+1]])

                    cnt += 1

                for ind_frames in list_of_ind:
                    n_interp = len(ind_frames)

                    # If missing frame is first or last then we just copy
                    if ind_frames[0] == 0:
                        print('not imp')
                    elif ind_frames[-1] == self.n_planes:
                        print('not imp')
                    else:
                        u1 = imread(os.path.join(folder, '{}_{}___{:06d}_{}c.tif'.format(a1, a2, ind_frames[0]-1, a3)))
                        u2 = imread(os.path.join(folder, '{}_{}___{:06d}_{}c.tif'.format(a1, a2, ind_frames[-1]+1, a3)))

                        for ii, ind in enumerate(ind_frames):
                            u = (ii+1)/(1+n_interp)*u1 + (1-(ii+1)/(1+n_interp))*u2
                            imsave(os.path.join(folder, '{}_{}___{:06d}_{}c.tif'.format(a1, a2, ind, a3)),
                                   u.astype('uint16'),
                                   check_contrast=False)

    @staticmethod
    def _is_valid_acquisition(path):
        """
        This function returns True if path folder is a CLEARSCOPE acquisition.
        Parameters
        ----------
        path: str
            path to check if acquisition is COLM

        Returns
        -------
        bool
        """
        return '0001' in os.listdir(path)

    @staticmethod
    def _get_n_channels(path):
        """
        Returns the number of channel of the CLEARSCOPE acquisition. This is based on a regex that checks
        the highest n in the path: some_path/X_Y___nx/
        """
        folders = glob(os.path.join(path, '0001', '*/'))
        return np.max([int(re.findall(r'__(\d)c', x)[0])+1 for x in folders])


# class mesospimParser(tileParser):
#     """
#     Class used to parse multi-tile colm data where each tile position in space matters. Tile parsed this way are usually
#     stitched later on.
#
#     """
#     def __init__(self, path, channel=0, verbose=True):
#         """
#         Constructor of the tileParser object for MESOSPIM acquisition.
#
#         Parameters
#         ----------
#         path: string
#             path where to look for the data. More specifically it should be the folder that contains the acquisition.
#         channel: int
#             fluorescence channel for parsing MESOSPIM data
#         verbose: bool
#             Control verbosity of the parsing. If True, the parser will print acquisition info in the terminal.
#
#         """
#
#         self.path = path
#         self.frame_size = 2048
#         self.channel = channel
#         self.channel_wavelength = self._get_channel_wavelength(channel)
#         self.type = 'mesospim'
#
#         self.tiles_list = self._get_tile_list()
#         self.n_tiles = len(self.tiles_list)
#         if self.n_tiles == 0:
#             raise FileNotFoundError('Error: no tile were found.')
#
#         # self._correct_offset()
#         # self.ncol = self._get_ncol()
#         # self.nrow = self._get_nrow()
#         # self._sort_tiles()
#         # self.tiles_pattern, self.tile_pattern_path = self._get_tiles_pattern()
#         # self.neighbors, self.n_edges = self._get_neighbors_map()
#         # self.neighbors_tot = self._get_total_neighbors_map()
#         # self.path_list = self._get_path_list()
#         # if verbose:
#         #     self._print_info()
#
#         # Define some folders
#         base, _ = os.path.split(self.path)
#         self.folder_root = base
#         self.folder_max_projs = os.path.join(base, 'max_projs')
#
#
#     def _get_channel_wavelength(self, channel):
#         files = glob(os.path.join(self.path, '*_Ch*_*.tiff'))
#         channel_list = []
#         for f in files:
#             pattern_search = re.findall(r'_Ch(\d{3})_', f)
#             if pattern_search != []:
#                 channel_list.append(int(pattern_search[-1]))
#
#         channel_list = sorted(list(set(channel_list)))
#
#         return int(channel_list[channel])
#     def _get_tiles_path(self):
#         """
#         Returns a list containing individual tiff.
#
#         """
#         return glob(os.path.join(self.path, '*_Ch{}_*.tiff'.format(self.channel_wavelength)))
#
#     def _get_tiles_from_path(self, files):
#         """
#         Returns a list of tiles as a dictionary.
#
#         """
#         tiles = []
#         for f in files:
#             # For each file we read the x and y position
#             x, y = get_pos()
#
#             # Then we have to find the row and col for each tile
#
#
#             tile = {'path': f,
#                     'row': row,
#                     'col': col,
#                     }
#             tiles.append(tile)
#         return tiles
#
#     def get_overlap(self):
#         """
#         Extract overlap from MESOSPIM `meta.txt` files.
#
#         """
#
#     @staticmethod
#     def _is_valid_acquisition(path):
#         """
#         This function returns True if path folder is a MESOSPIM acquisition.
#         Parameters
#         ----------
#         path: str
#             path to check if acquisition is COLM
#
#         Returns
#         -------
#         bool
#         """
#         return glob(path, '*_meta.txt') != 0
#
#     @staticmethod
#     def _get_n_channels(path):
#         """
#         Returns the number of channel of the MESOSPIM acquisition.
#         """
#         files = sorted(glob(os.path.join(path, '*Tile0*_*.tif')))
#         return len(files)