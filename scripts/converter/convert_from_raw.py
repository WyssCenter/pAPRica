"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""
import numpy as np
import pyapr
from glob import glob
import os
from skimage.io import imread
import pipapr

# Parameters
data_path = '/run/user/1000/gvfs/smb-share:server=fcbgnasc.campusbiotech.ch,share=fcbgdata/0063_CBT_UNIGE_LAMY/Tomas Jorda/mesoSpim/20210609/'

tiles = pipapr.parser.randomParser(data_path, frame_size=2048, ftype='raw')

converter = pipapr.converter.tileConverter(tiles)
converter.set_compression(quantization_factor=1, bg=190)
converter.batch_convert(Ip_th=190)