"""
Script to benchmark the new maximum projection algorithm.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
import pyapr
from glob import glob
import pyapr
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
dim = 2
path = '/home/apr-benchmark/Desktop/data/sarah/APR/0_0.apr'
patch = pyapr.ReconPatch()
patch.x_begin = 128
patch.x_end = 1024
patch.y_begin = 128
patch.y_end = 1024
patch.z_begin = 128
patch.z_end = 1024

apr = pyapr.APR()
parts = pyapr.ShortParticles()
pyapr.io.read(path, apr, parts)
t = time()
for i in range(10):
    alt = pyapr.numerics.transform.maximum_projection_alt_patch(apr, parts, dim=dim, patch=patch)
print('Elapsed time alt: {} s.'.format(time()-t))
t = time()
for i in range(10):
    old = pyapr.numerics.transform.maximum_projection_patch(apr, parts, dim=dim, patch=patch)
print('Elapsed time old: {} s.'.format(time()-t))

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].imshow(np.log(old+1), vmin=0, vmax=10, cmap='gray')
ax[0].set_title('Old')
ax[1].imshow(np.log(alt+1), vmin=0, vmax=10, cmap='gray')
ax[1].set_title('Alt')

print(np.sum(alt==old)/alt.size*100)