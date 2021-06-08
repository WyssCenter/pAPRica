"""
Module containing classes and functions relative to Converting.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
import pyapr
import os
from pathlib import Path
from alive_progress import alive_bar

class tileConverter():

    def __init__(self,
                 tiles: pipapr.parser.tileParser):

        self.tiles = tiles
        self.path = tiles.path
        self.n_tiles = tiles.n_tiles

        self.compression = 0
        self.bg = None
        self.quantization_factor = None

    def set_compression(self, quantization_factor=1, bg=108):

        self.compression = 1
        self.bg = bg
        self.quantization_factor = quantization_factor


    def batch_convert(self, Ip_th=108, rel_error=0.2, gradient_smoothing=2, dx=1, dy=1, dz=1):

        # Safely create folder to save apr data
        base_folder, _ = os.path.split(self.path)
        folder_apr = os.path.join(base_folder, 'APR')
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
                apr, parts = pyapr.converter.get_apr(tile.data, params=par)

                if self.compression:
                    parts.set_compression_type(1)
                    parts.set_quantization_factor(self.quantization_factor)
                    parts.set_background(self.bg)

                filename = '{}_{}.apr'.format(tile.row, tile.col)
                pyapr.io.write(os.path.join(folder_apr, filename), apr, parts)
                bar()

        # Modify tileParser object to use APR instead
        self.tiles = pipapr.parser.tileParser(folder_apr,
                                              frame_size=self.tiles.frame_size,
                                              overlap=self.tiles.overlap,
                                              ftype='apr')