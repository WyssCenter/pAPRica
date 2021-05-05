"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import os
import pyapr
import numpy as np
from sklearn.cluster import KMeans
import napari
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_gradients(apr, parts, sobel=True):
    par = apr.get_parameters()
    dx = pyapr.FloatParticles()
    dy = pyapr.FloatParticles()
    dz = pyapr.FloatParticles()

    pyapr.numerics.gradient(apr, parts, dz, dimension=2, delta=par.dz, sobel=sobel)
    pyapr.numerics.gradient(apr, parts, dx, dimension=1, delta=par.dx, sobel=sobel)
    pyapr.numerics.gradient(apr, parts, dy, dimension=0, delta=par.dy, sobel=sobel)
    return dz, dx, dy


def compute_laplacian(apr, parts, sobel=True):
    par = apr.get_parameters()
    dz, dx, dy = compute_gradients(apr, parts, sobel)
    dx2 = pyapr.FloatParticles()
    dy2 = pyapr.FloatParticles()
    dz2 = pyapr.FloatParticles()
    pyapr.numerics.gradient(apr, dz, dz2, dimension=2, delta=par.dz, sobel=sobel)
    pyapr.numerics.gradient(apr, dx, dx2, dimension=1, delta=par.dx, sobel=sobel)
    pyapr.numerics.gradient(apr, dy, dy2, dimension=0, delta=par.dy, sobel=sobel)
    return dz + dx + dy


def compute_gradmag(apr, parts, sobel=True):
    par = apr.get_parameters()
    gradmag = pyapr.FloatParticles()
    pyapr.numerics.gradient_magnitude(apr, parts, gradmag, deltas=(par.dz, par.dx, par.dy), sobel=True)
    return gradmag


def gaussian_blur(apr, parts, sigma=1.5, size=11):
    stencil = pyapr.numerics.get_gaussian_stencil(size, sigma, 3, True)
    output = pyapr.FloatParticles()
    pyapr.numerics.filter.convolve_pencil(apr, parts, output, stencil, use_stencil_downsample=True,
                                          normalize_stencil=True, use_reflective_boundary=True)
    return output


def particle_levels(apr, normalize=True):
    lvls = pyapr.ShortParticles(apr.total_number_particles())
    lvls.fill_with_levels(apr)
    if normalize:
        lvls *= (1 / apr.level_max())
    return lvls


def compute_std(apr, parts, size=5):
    dims = apr.org_dims()
    box_size = [size if d >= size else 1 for d in dims]
    locstd = pyapr.FloatParticles()
    pyapr.numerics.local_std(apr, parts, locstd, size=box_size)
    return locstd


def display_segmentation(u, lmap, vdim=2):
    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=vdim)
        # add the volume
        viewer.add_image(u, name='Intensity image')
        # add labels
        viewer.add_labels(lmap, name='segmentation')

def scatter_plot_3D(y, x, subsample=1000):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot particle belonging to 1st cluster
    ind_0 = (y==0)
    x_0 = x[ind_0]
    ax.scatter(x_0[::subsample, 0], x_0[::subsample, 1], x_0[::subsample, 2], marker='o', label='0')

    # Plot particle belonging to 2nd cluster
    ind_1 = (y==1)
    x_1 = x[ind_1]
    ax.scatter(x_1[::subsample, 0], x_1[::subsample, 1], x_1[::subsample, 2], marker='o', label='1')

    plt.legend()
    ax.set_xlabel('Gradient magnitude')
    ax.set_ylabel('Smoothed intensity')
    ax.set_zlabel('Diff. of Gaussian')


def scatter_features_2D(y, x, x_names=None, subsample=1000):
    """
    This function plots each features again each others.
    The total number of plots is therefore n_features*(n_features-1)/2 so condider no using this function
    with high number of features:

    n_features - n_plots
        3      -    3
        4      -    6
        5      -    10
        6      -    15

    Parameters
    ----------
    y: (array) labels for coloring the markers with respect to their class
    x: (array) features with shape (n_sample, n_features)
    x_names: (list, optional) list containing the name of features for x/y_labels
    subsample: (int) subsampling factor for plotting

    Returns
    -------
    None
    """
    y = np.array(y, copy=True)
    n_features = x.shape[1]

    for i in range(n_features-1):
        for ii in range(i+1, n_features):
            plt.figure()
            plt.scatter(x[::subsample, i], x[::subsample, ii], c=y[::subsample])
            if x_names is not None:
                plt.xlabel(x_names[i])
                plt.ylabel(x_names[ii])


# Parameters
data_dir = '/media/sf_shared_folder_virtualbox/mouse_2P/data1'
fpath_apr = os.path.join(data_dir, '2P_mouse_re0.2.apr')
parts = pyapr.ShortParticles()

# read APR
apr = pyapr.APR()
pyapr.io.read(fpath_apr, apr, parts)

# compute features
start = time()
levels = particle_levels(apr)
gradmag = compute_gradmag(apr, parts)
smooth = gaussian_blur(apr, parts, sigma=0.7, size=5)
dog = smooth - gaussian_blur(apr, parts, sigma=2.0, size=13)
lapl = compute_laplacian(apr, parts)
print('compute features took {} seconds'.format(time()-start))

# Construct feature matrix
features = [gradmag, smooth, dog]
x_names = ['Gradient magnitude', 'Smoothed intensity', 'Diff. of Gaussian']
X = np.empty((apr.total_number_particles(), len(features)), dtype=np.float32)
for i, x in enumerate(features):
    X[:, i] = np.array(features[i])

# Run K-means clustering
start = time()
clf = KMeans(n_clusters=2, random_state=1337)
clf.fit(X)
print('kmeans took {} seconds'.format(time()-start))

# Copy labels to ParticleData
mask = pyapr.ShortParticles()
mask.resize(apr.total_number_particles())
if clf.labels_[0] == 1 and clf.labels_.max() == 1:  # first particle is most likely background
    for i, x in enumerate(clf.labels_):
        mask[i] = 1-x
else:
    for i, x in enumerate(clf.labels_):
        mask[i] = x

# Compute connected component
cc = pyapr.ShortParticles()
pyapr.numerics.segmentation.connected_component(apr, mask, cc)

# Display result
cc_pix = np.array(pyapr.numerics.reconstruction.recon_pc(apr, cc), copy=True).astype(np.uint16)
u = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=True)
display_segmentation(u, cc_pix)
scatter_features_2D(mask, X, x_names)