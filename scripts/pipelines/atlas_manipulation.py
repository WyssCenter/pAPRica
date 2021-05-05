"""
This script takes the registered atlas, compress it and allows to find the ID of any pixel on
the full resolution space.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from skimage.io import imread
import pyapr
import numpy as np
import matplotlib.pyplot as plt

# Parameters
atlas_path = r'/mnt/Data/wholebrain/brainreg_output_autofluo_clahe_crop/registered_atlas.tiff'

# Load atlas tiff
atlas = imread(atlas_path)

# Convert atlas to APR (not really needed as its size should be small because of the down-sampling).
# CAREFUL there is a loss of information due to casting uint32 to uint16 because of the weird
# Allen brain numbering convention. To alleviate this, a hashtable can be used.
apr = pyapr.APR()
parts = pyapr.FloatParticles()
par = pyapr.APRParameters()
par.rel_error = 0.00001
par.gradient_smoothing = 0
par.grad_th = 0.001
par.sigma_th = 0.001
par.Ip_th = 0
apr, parts = pyapr.converter.get_apr(atlas.astype('uint16'), params=par)

# Visualize artifacts created by the uint16 casting
toto = pyapr.numerics.reconstruction.reconstruct_constant(apr, parts)
print(apr.computational_ratio())
print(np.sum(toto!=atlas)/toto.size*100)
i = 250
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
ax[0].imshow(toto[i]!=atlas[i])
ax[1].imshow(atlas[i])
ax[1].set_title('ATLAS')
ax[2].imshow(toto[i])
ax[2].set_title('APR')