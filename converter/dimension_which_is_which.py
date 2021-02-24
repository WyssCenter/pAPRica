from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import pyapr

u = imread(r'/media/sf_shared_folder_virtualbox/terastitcher_tests/000000/000000_000000/000000.tif')

proj_pix = []
for i in range(3):
    proj_pix.append(np.max(u, axis=i))

par = pyapr.APRParameters()
par.auto_parameters = False  # really heuristic and not working
par.sigma_th = 26.0
par.grad_th = 3.0
par.Ip_th = 253.0
par.rel_error = 0.2
par.gradient_smoothing = 2

# Convert data to APR
apr, parts = pyapr.converter.get_apr(image=u, params=par, verbose=False)
proj_apr = []
for i in range(3):
    proj_apr.append(pyapr.numerics.transform.maximum_projection(apr, parts, dim=i))

fig, ax = plt.subplots(2, 3)
for i in range(3):
    ax[0, i].imshow(proj_pix[i], cmap='gray')
    ax[1, i].imshow(proj_apr[i], cmap='gray')