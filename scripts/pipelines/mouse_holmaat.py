"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
from time import time
import napari
import pandas as pd


path = '/run/user/1000/gvfs/smb-share:server=wyssnasc.campusbiotech.ch,share=computingdata/Alice/8004_CBT_WYSS_HBM/Jules/HOLT_011398_LOC000_20210721_170347/VW0/'

tiles = pipapr.parser.tileParser(path, overlap=28, frame_size=2048, ncol=7, nrow=7, ftype='tiff2D')
converter = pipapr.converter.tileConverter(tiles)
# converter.set_compression(bg=1000)
converter.batch_convert_to_apr(Ip_th=120, rel_error=0.2, gradient_smoothing=0, path='/home/hbm/Desktop/data/holmaat')