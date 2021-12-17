"""
Script to test single tile atlasing.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
import pyapr

# Parameters
apr_path = '/home/apr-benchmark/Desktop/data/sarah/APR/0_0.apr'

# Load data
apr = pyapr.APR()
parts = pyapr.ShortParticles()
pyapr.io.read(apr_path, apr, parts)

# Down-sample and reconstruct data for Brainreg
slicer = pyapr.data_containers.APRSlicer(apr, parts, level_delta=-2)
data = slicer[:, :, :]

# Atlas data
atlaser = pipapr.atlaser.tileAtlaser(original_pixel_size=[5, 5.26, 5.26],
                                     downsample=4,
                                     atlas=None,
                                     merged_data=data)
atlaser.register_to_atlas(output_dir='/home/apr-benchmark/Desktop/data/sarah/APR',
                          orientation='ipr')
