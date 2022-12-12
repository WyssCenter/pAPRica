"""
Script to test single tile atlasing.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import paprica
import pyapr
import os


def main():
    # Parameters
    apr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'apr', '0_2.apr')

    # Load data
    apr, parts = pyapr.io.read(apr_path)

    # Down-sample and reconstruct data for Brainreg
    slicer = pyapr.reconstruction.APRSlicer(apr, parts, level_delta=-2)
    data = slicer[:, :, :]

    # Atlas data
    atlaser = paprica.atlaser.tileAtlaser(original_pixel_size=[5, 5.26, 5.26],
                                          downsample=4,
                                          atlas=None,
                                          merged_data=data)
    atlaser.register_to_atlas(orientation='ipr')


if __name__ == '__main__':
  main()