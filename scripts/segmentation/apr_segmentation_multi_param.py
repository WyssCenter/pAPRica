"""
This script compares the segmentation perf. using the same classifier but on APR computed with different parameters.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pyapr
import numpy as np
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
import napari
from time import time
import os


def are_labels_the_same(local_labels):
    labels = local_labels[local_labels != 0].flatten()
    return ((labels == labels[0]).all(), labels[0])


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


def predict_on_APR(clf, x):
    # Predict on numpy array
    t = time()
    y_pred = clf.predict(x)
    print('Prediction took {} s.\n'.format(time()-t))

    # Transform numpy array to ParticleData
    parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

    return parts_pred


def predict_and_display(apr, parts, clf, x, save_results=None):
    data = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)
    t = time()
    y_pred = clf.predict(x)
    print('Prediction took {} s.\n'.format(time()-t))
    mask = pyapr.ShortParticles(apr.total_number_particles())
    for i, elem in enumerate(y_pred):
        mask[i] = elem
    labels = np.array(pyapr.numerics.reconstruction.recon_pc(apr, mask), copy=False)
    if save_results is not None:
        imsave(save_results, labels.astype('uint8'), check_contrast=False)
    display_segmentation(data, labels, vdim=2)
    return y_pred


def sample_labels_on_APR(labels, apr, parts_train):

    apr_it = apr.iterator()
    org_dims = apr.org_dims()

    for z in range(apr_it.z_num(apr.level_max())):
        for x in range(apr_it.x_num(apr.level_max())):
            for idx in range(apr_it.begin(apr.level_max(), z, x), apr_it.end()):
                parts_train[idx] = labels[z, x, apr_it.y(idx)]

    # loop over levels up to level_max-1 to assign the remaining labels
    for level in range(apr_it.level_min(), apr_it.level_max()):
        step_size = 2 ** (apr.level_max() - level)  # this is the size (in pixels) of the particle cells at level
        for z in range(apr_it.z_num(level)):
            for x in range(apr_it.x_num(level)):
                for idx in range(apr_it.begin(level, z, x), apr_it.end()):
                    y = apr_it.y(idx)

                    y_start = y * step_size
                    x_start = x * step_size
                    z_start = z * step_size

                    y_end = min(y_start + step_size, org_dims[0])
                    x_end = min(x_start + step_size, org_dims[1])
                    z_end = min(z_start + step_size, org_dims[2])

                    local_labels = labels[z_start:z_end, x_start:x_end, y_start:y_end]

                    if (local_labels == 0).all():  # case where there is no label
                        parts_train[idx] = 0
                    elif np.sum(local_labels != 0) == 1:  # case where there is only one label
                        parts_train[idx] = local_labels.max()
                    else:  # case where there are several labels
                        same, l = are_labels_the_same(local_labels)
                        if same:
                            parts_train[idx] = l
                        else:
                            parts_train[idx] = 0
                            print('Ambiguous label detected, set it to 0.')
    return parts_train


# APR file to segment
# Parameters
folder_path = r'/media/sf_shared_folder_virtualbox/PV_interneurons'
image_path = r'substack.tif'
fpath_labels = r'/media/sf_shared_folder_virtualbox/PV_interneurons/manual_sparse_labels_membrane.npy'

# Load image and labels
data = imread(os.path.join(folder_path, image_path))
data = data[:, :, 800:800+512]
labels = np.load(fpath_labels).squeeze().astype('uint16') # 0: not labeled - 1: cells - 2: background - 3: membrane
labels = labels[:, :, 800:800+512]

x = []
y = []
f_list = []
apr_list = []
parts_list = []
for i, rel_error in enumerate([0.1, 0.2, 0.4]):
    par = pyapr.APRParameters()
    par.auto_parameters = False # really heuristic and not working
    par.sigma_th = 562.0
    par.grad_th = 49.0
    par.Ip_th = 903.0
    par.rel_error = rel_error
    par.gradient_smoothing = 2.0
    par.dx = 1
    par.dy = 1
    par.dz = 3
    apr, parts = pyapr.converter.get_apr(data, verbose=True, params=par)
    apr_list.append(apr)
    parts_list.append(parts)
    print('CR: {:.2f}'.format(data.size/apr.total_number_particles()))

    # Sample labels on APR grid
    parts_train_C = pyapr.ShortParticles(apr.total_number_particles())
    parts_train_C = sample_labels_on_APR(labels, apr, parts_train_C)
    parts_train = np.array(parts_train_C, copy=False)

    t = time()
    # Compute gradient along a dimension (Sobel filter). dimension can be 0, 1 or 2
    grad_x, grad_y, grad_z = compute_gradients(apr, parts)

    # Compute gradient magnitude (central finite differences)
    grad = compute_gradmag(apr, parts)

    # Compute local standard deviation around each particle
    local_std = compute_std(apr, parts, size=5)

    # Compute lvl for each particle
    lvl = particle_levels(apr, normalize=True)

    # Compute difference of Gaussian
    dog = gaussian_blur(apr, parts, sigma=3, size=22) - gaussian_blur(apr, parts, sigma=1.5, size=11)
    print('Features computation took {} s.'.format(time()-t))

    # Aggregate filters in a feature array
    f = np.vstack((np.array(parts, copy=True),
                   lvl,
                   grad_x,
                   grad_y,
                   grad_z,
                   grad,
                   local_std,
                   dog
                   )).T
    f_list.append(f)
    f_names = ['Intensity',
                   'lvl',
                   'grad_x',
                   'grad_y',
                   'grad_z',
                   'grad_mag',
                   'local_std',
                   'dog'
               ]

    # Fetch data that was manually labelled
    ind_manual_label = (parts_train != 0)
    if i == 0:
        x = f[ind_manual_label, :]
        y = parts_train[ind_manual_label]-1
    else:
        x = np.append(x, f[ind_manual_label, :], axis=0)
        y = np.append(y, parts_train[ind_manual_label]-1)

# Train SVM classification
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

clf = make_pipeline(preprocessing.StandardScaler(with_mean=True, with_std=True),
                    RandomForestClassifier(n_estimators=10, class_weight='balanced'))
t = time()
clf.fit(x, y)
print('Training took {} s.\n'.format(time()-t))

# Training accuracy:
y_pred = clf.predict(x)
print('Training accuracy: {:.2f}%.'.format(np.sum((y_pred-y==0))/y.size*100))

# Apply on whole dataset and display results
n = 1
f = f_list[n]
apr = apr_list[n]
parts = parts_list[n]
# Predict particle type (cell, membrane or brackground) for each cell with the trained model
parts_pred = predict_on_APR(clf, f)
# Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
parts_cells = (parts_pred == 0)
# Remove small holes to get the misclassified nuclei
parts_cells = pyapr.numerics.transform.remove_small_holes(apr, parts_cells, min_volume=500)
# Opening to better separate touching cells
pyapr.numerics.transform.opening(apr, parts_cells, radius=1, inplace=True)
# Apply connected component
cc = pyapr.ShortParticles()
pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)
# Remove small objects
pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=200)
pyapr.numerics.transform.dilation(apr, cc, radius=1)

# Display results with napari
lmap = np.array(pyapr.numerics.reconstruction.recon_pc(apr, cc), copy=False)
data = np.array(pyapr.numerics.reconstruction.recon_smooth(apr, parts), copy=False)
display_segmentation(data, lmap)

# Save classifier
# Note: this method is not perfect and it might break if scikit-learn version is not the same between the
# dump and the loading. Use with caution (see https://scikit-learn.org/stable/modules/model_persistence.html)
# from joblib import dump
# dump(clf, '/media/sf_shared_folder_virtualbox/mouse_2P/data1/classifiers/random_forest_n100.joblib')
# To load back use:
# from joblib import load
# toto = load('/media/sf_shared_folder_virtualbox/mouse_2P/data1/classifiers/random_forest_n100.joblib')

# Get labeled objects patch coordinates (some id do not appear in cc)
# minc, maxc = pyapr.numerics.transform.find_objects(apr, cc)
# valid_labels = [(minc[i, :] < maxc[i, :]).all() for i in range(minc.shape[0])]
# minc = minc[valid_labels, :]
# maxc = maxc[valid_labels, :]
#
# # Compute the number of detected cells
# print('Total number of detected cells: {}\n'.format(len(valid_labels)))

