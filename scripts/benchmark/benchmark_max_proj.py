"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
from glob import glob
import pyapr
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
path = '/home/apr-benchmark/Desktop/data/synthetic/APR'

folders = glob(os.path.join(path, '*/'))
cr = []
time_mp_0 = []
time_mp_1 = []
time_mp_2 = []
for folder in folders:
    tiles = pipapr.parser.randomParser(folder, frame_size=512, ftype='apr')
    tmp_cr = []
    tmp_0 = 0
    tmp_1 = 0
    tmp_2 = 0
    for tile in tiles:
        tile.load_tile()

        # DIM 0
        t = time()
        pyapr.numerics.transform.maximum_projection_alt(tile.apr, tile.parts, dim=0)
        tmp_0 += time() - t
        # DIM 1
        t = time()
        pyapr.numerics.transform.maximum_projection_alt(tile.apr, tile.parts, dim=1)
        tmp_1 += time() - t
        # DIM 2
        t = time()
        pyapr.numerics.transform.maximum_projection_alt(tile.apr, tile.parts, dim=2)
        tmp_2 += time() - t

        tmp_cr.append(tile.apr.computational_ratio())

    cr.append(np.mean(tmp_cr))
    time_mp_0.append(tmp_0)
    time_mp_1.append(tmp_1)
    time_mp_2.append(tmp_2)

    print('CR {} - Elapsed time 0: {}, 1: {}, 2: {} s.'.format(cr[-1], time_mp_0[-1], time_mp_1[-1], time_mp_2[-1]))


# Display results
plt.figure(1)
plt.plot(cr, np.array(time_mp_0)/16, 'k+', label='Dim 0')
plt.plot(cr, np.array(time_mp_1)/16, 'r+', label='Dim 1')
plt.plot(cr, np.array(time_mp_2)/16, 'b+', label='Dim 2')
plt.xlabel('Computational ratio')
plt.ylabel('Max proj time [s.]')
plt.title('Time to max project a 2048x512x512 tile')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.001, 2])
plt.savefig('n_cores_{}_alt.png'.format(os.environ['OMP_NUM_THREADS']))