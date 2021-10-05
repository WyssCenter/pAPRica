"""
Submodule containing classes and functions relative to **converting** tiles to APR.

The general workflow is first to parse tiles using a *parser* object and then convert using the tileConverter class.
This class is essentially a wrapper to pylibapr which allows to facilitate batch conversions and batch reconstructions.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
import pyapr
import os
from pathlib import Path
from alive_progress import alive_bar
from skimage.io import imsave

class tileConverter():
    """
    Class to convert tiles to APR or to tiff.
    """

    def __init__(self,
                 tiles: (pipapr.parser.tileParser, pipapr.parser.randomParser)):
        """

        Parameters
        ----------
        tiles: (tileParser or randomParser) parser object referencing tiles to be converted.
        """

        if isinstance(tiles, pipapr.parser.randomParser):
            self.random = True # Not multitile
        else:
            self.random = False # Multitile

        self.tiles = tiles
        self.path = tiles.path
        self.n_tiles = tiles.n_tiles

        self.compression = 0
        self.bg = None
        self.quantization_factor = None

    def set_compression(self, quantization_factor=1, bg=108):
        """
        Activate B3D compression for saving tiles.

        Parameters
        ----------
        quantization_factor: (int) quantization factor: the higher, the more compressed
                                (refer to B3D paper for more detail).
        bg: (int) background value: any value below this threshold will be set to the background value. This helps
                save up space by having the same value for the background (refer to B3D paper for more details).

        Returns
        -------
        None
        """

        self.compression = True
        self.bg = bg
        self.quantization_factor = quantization_factor

    def deactivate_compression(self):
        """
        Deactivate B3D compression when saving particles.

        Returns
        -------
        None
        """

        self.compression = False
        self.bg = None
        self.quantization_factor = None

    def batch_convert_to_apr(self,
                             Ip_th=108,
                             rel_error=0.2,
                             gradient_smoothing=2,
                             dx=1,
                             dy=1,
                             dz=1,
                             path=None,
                             lazy_loading=True):
        """
        Convert all parsed tiles to APR using auto-parameters.

        Parameters
        ----------
        Ip_th: (int) Intensity threshold
        rel_error: (float in [0, 1[) relative error bound
        gradient_smoothing: (float) B-Spline smoothing parameter (typically between 0 (no smoothing) and 10 (LOTS of
                            smoothing)
        dx: (float) PSF size in x, used to compute the gradient
        dy: (float) PSF size in y, used to compute the gradient
        dz: (float) PSF size in z, used to compute the gradient
        lazy_loading: (bool) if lazy_loading is true then the converter save mean tree particle which are necessary
                            for lazy loading of the APR. It will require about 1/7 more storage.

        Returns
        -------
        None
        """

        if self.tiles.type == 'apr':
            raise TypeError('Error: data already in APR format.')

        # Safely create folder to save apr data
        if path is None:
            base_folder, _ = os.path.split(self.path)
            folder_apr = os.path.join(base_folder, 'APR')
        else:
            folder_apr = path
        Path(folder_apr).mkdir(parents=True, exist_ok=True)

        with alive_bar(total=self.n_tiles, title='Converting tiles', force_tty=True) as bar:
            for tile in self.tiles:
                tile.load_tile()

                # Set parameters
                par = pyapr.APRParameters()
                par.Ip_th = Ip_th
                par.rel_error = rel_error
                par.dx = dx
                par.dy = dy
                par.dz = dz
                par.gradient_smoothing = gradient_smoothing
                par.auto_parameters = True

                # Convert tile to APR and save
                apr = pyapr.APR()
                parts = pyapr.ShortParticles()
                converter = pyapr.converter.ShortConverter()
                converter.set_parameters(par)
                converter.verbose = True
                converter.get_apr(apr, tile.data)
                parts.sample_image(apr, tile.data)

                if self.compression:
                    parts.set_compression_type(1)
                    parts.set_quantization_factor(self.quantization_factor)
                    parts.set_background(self.bg)

                if lazy_loading:
                    tree_parts = pyapr.ShortParticles()
                    pyapr.numerics.fill_tree_mean(apr, parts, tree_parts)
                else:
                    tree_parts = None

                # Save converted data
                if self.random:
                    basename, filename = os.path.split(tile.path)
                    pyapr.io.write(os.path.join(folder_apr, filename[:-4] + '.apr'), apr, parts, tree_parts=tree_parts)
                else:
                    filename = '{}_{}.apr'.format(tile.row, tile.col)
                    pyapr.io.write(os.path.join(folder_apr, filename), apr, parts, tree_parts=tree_parts)

                bar()

        if not self.random:
            # Modify tileParser object to use APR instead
            self.tiles = pipapr.parser.tileParser(folder_apr,
                                                  frame_size=self.tiles.frame_size,
                                                  ftype='apr')

    def batch_reconstruct_pixel(self, mode='constant'):
        """
        Reconstruct all APR tiles to pixel data.

        Parameters
        ----------
        mode: (str) reconstruction mode, can be 'constant', 'smooth' or 'level'

        Returns
        -------
        None
        """

        if self.tiles.type != 'apr':
            raise TypeError('Error: data not in APR format.')

        # Safely create folder to save apr data
        base_folder, _ = os.path.split(self.path)
        folder_tiff = os.path.join(base_folder, 'TIFF')
        Path(folder_tiff).mkdir(parents=True, exist_ok=True)

        with alive_bar(total=self.n_tiles, title='Converting tiles', force_tty=True) as bar:
            for tile in self.tiles:
                tile.load_tile()

                if mode == 'constant':
                    data = pyapr.numerics.reconstruction.reconstruct_constant(tile.apr, tile.parts).squeeze()
                elif mode=='smoth':
                    data = pyapr.numerics.reconstruction.reconstruct_smooth(tile.apr, tile.parts).squeeze()
                elif mode=='level':
                    data = pyapr.numerics.reconstruction.reconstruct_level(tile.apr, tile.parts).squeeze()
                else:
                    raise ValueError('Error: unknown mode for APR reconstruction.')

                # Save converted data
                if self.random:
                    basename, filename = os.path.split(tile.path)
                    imsave(os.path.join(folder_tiff, filename[:-4] + '.tif'), data, check_contrast=False)
                else:
                    filename = '{}_{}.tif'.format(tile.row, tile.col)
                    imsave(os.path.join(folder_tiff, filename), data, check_contrast=False)

                bar()



