"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""
import numpy as np
import pyapr
from glob import glob
import os
from skimage.io import imread

def load_raw(path):
    u = np.fromfile(path, dtype='uint16', count=-1)
    return u.reshape((-1, 2048, 2048))

# Parameters
data_path = r'/mnt/Data/wholebrain/multitile'
compress = False
par = pyapr.APRParameters()
par.rel_error = 0.2
par.gradient_smoothing = 2
par.dx = 1
par.dy = 1
par.dz = 1
# par.Ip_th = 700.0
# par.sigma_th = 200.0
# par.grad_th = 65.0
par.Ip_th = 220
par.sigma_th = 50.0
par.grad_th = 3.2

files = glob(os.path.join(data_path, '*raw'))
for f in files:

    apr, parts = pyapr.converter.get_apr(load_raw(f), params=par)

    if compress:
        parts.set_compression_type(1)
        parts.set_quantization_factor(2)
        parts.set_background(125)

    pyapr.io.write(f[:-4] + '_no_compression.apr', apr, parts)

    if compress:
        # Size of original and compressed APR files in MB
        tiff_file_size = os.path.getsize(f) * 1e-6
        compressed_file_size = os.path.getsize(f[:-4] + '.apr') * 1e-6

        print("APR CR: {:7.2f}".format(apr.computational_ratio()))
        print("Lossy Compressed APR File Size: {:7.2f} MB".format(compressed_file_size))
        print("Lossy Memory Compression Ratio: {:7.2f} ".format(tiff_file_size/compressed_file_size))

