"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pyapr
import numpy as np
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
import napari
from time import time


def are_labels_the_same(local_labels):
    """
    Determine if manual labels in particle are the same and return the labels

    Parameters
    ----------
    local_labels: (array) particle labels

    """
    labels = local_labels[local_labels != 0].flatten()
    return ((labels == labels[0]).all(), labels[0])


def compute_gradients(apr, parts, sobel=True):
    """
    Compute gradient for each spatial direction directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    sobel: (bool) use sobel filter to compute the gradient

    Returns
    -------
    (dx, dy, dz): (arrays) gradient for each direction
    """

    par = apr.get_parameters()
    dx = pyapr.FloatParticles()
    dy = pyapr.FloatParticles()
    dz = pyapr.FloatParticles()

    pyapr.numerics.gradient(apr, parts, dz, dimension=2, delta=par.dz, sobel=sobel)
    pyapr.numerics.gradient(apr, parts, dx, dimension=1, delta=par.dx, sobel=sobel)
    pyapr.numerics.gradient(apr, parts, dy, dimension=0, delta=par.dy, sobel=sobel)
    return dz, dx, dy


def compute_laplacian(apr, parts, grad=None, sobel=True):
    """
    Compute Laplacian for each spatial direction directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    grad: (dz, dy, dx) gradient for each direction if precomputed (faster for Laplacian computation)
    sobel: (bool) use sobel filter to compute the gradient

    Returns
    -------
    Laplacian of APR.
    """

    par = apr.get_parameters()
    if grad is None:
        dz, dx, dy = compute_gradients(apr, parts, sobel)
    else:
        dz, dx, dy = grad
    dx2 = pyapr.FloatParticles()
    dy2 = pyapr.FloatParticles()
    dz2 = pyapr.FloatParticles()
    pyapr.numerics.gradient(apr, dz, dz2, dimension=2, delta=par.dz, sobel=sobel)
    pyapr.numerics.gradient(apr, dx, dx2, dimension=1, delta=par.dx, sobel=sobel)
    pyapr.numerics.gradient(apr, dy, dy2, dimension=0, delta=par.dy, sobel=sobel)
    return dz2 + dx2 + dy2


def compute_gradmag(apr, parts, sobel=True):
    """
    Compute gradient magnitude directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    sobel: (bool) use sobel filter to compute the gradient

    Returns
    -------
    Gradient magnitude of APR.
    """

    par = apr.get_parameters()
    gradmag = pyapr.FloatParticles()
    pyapr.numerics.gradient_magnitude(apr, parts, gradmag, deltas=(par.dz, par.dx, par.dy), sobel=sobel)
    return gradmag


def gaussian_blur(apr, parts, sigma=1.5, size=11):
    """
    Compute Gaussian blur directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    sigma: (float) Gaussian blur standard deviation (kernel radius)
    size: (int) kernel size (increase with caution, complexity is not linear)

    Returns
    -------
    Blurred APR.
    """

    stencil = pyapr.numerics.get_gaussian_stencil(size, sigma, ndims=3, normalize=True)
    output = pyapr.FloatParticles()
    pyapr.numerics.filter.convolve_pencil(apr, parts, output, stencil, use_stencil_downsample=True,
                                          normalize_stencil=True, use_reflective_boundary=True)
    return output


def particle_levels(apr):
    """
    Returns apr level: for each particle the lvl is defined as the size of the particle in pixel.

    Parameters
    ----------
    apr: (APR) APR object

    Returns
    -------
    Particle level.
    """

    lvls = pyapr.ShortParticles(apr.total_number_particles())
    lvls.fill_with_levels(apr)
    lvls = np.array(lvls)

    return 2 ** (lvls.max() - lvls)


def compute_std(apr, parts, size=5):
    """
    Compute local standard deviation directly on APR.

    Parameters
    ----------
    apr: (APR) APR object
    parts: (ParticleData) particle data sampled on APR
    size: (int) kernel size

    Returns
    -------
    Local standard deviation of APR.
    """

    dims = apr.org_dims()
    box_size = [size if d >= size else 1 for d in dims]
    locstd = pyapr.FloatParticles()
    pyapr.numerics.local_std(apr, parts, locstd, size=box_size)
    return locstd


def predict_on_APR_block(clf, x, n_parts=1e7):
    # Predict on numpy array by block to avoid memory issues
    t = time()
    y_pred = np.empty((x.shape[0]))
    n_block = int(np.ceil(x.shape[0]/n_parts))
    if int(n_parts) != n_parts:
        raise ValueError('Error: n_parts must be an int.')
    n_parts = int(n_parts)

    clf[1].set_params(n_jobs=-1)
    for i in range(n_block):
        y_pred[i*n_parts:min((i+1)*n_parts, x.shape[0])] = clf.predict(x[i*n_parts:min((i+1)*n_parts, x.shape[0])])

    print('Blocked prediction took {} s.\n'.format(time()-t))

    # Transform numpy array to ParticleData
    parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

    return parts_pred


def sample_labels_on_APR(labels, apr, parts_train):

    apr_it = apr.iterator()

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

                    y_end = min(y_start + step_size, apr.org_dims(0))
                    x_end = min(x_start + step_size, apr.org_dims(1))
                    z_end = min(z_start + step_size, apr.org_dims(2))

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

def analyze_labels(lvl, label):

    label_unique = np.unique(label)

    fig, ax = plt.subplots(1, label_unique.size)
    for i, l in enumerate(label_unique):
        lvl_l = lvl[label == l]
        ax[i].hist(lvl_l, bins=np.unique(lvl_l), log=True, label='Class {}'.format(l))
        ax[i].set_xlabel('Particle size [pixel]')
        ax[i].legend()

def filter_manual_labels(ind, label, lvl):

    ind_filtered = []
    label_filtered = []
    cnt = 0
    # Remove cell label if particle is too big
    for i in np.where(label == 0)[0]:
        if lvl[i] < 2:
            ind_filtered.append(ind[i])
            label_filtered.append(label[i])
        else:
            cnt += 1
    # Remove membrane label if particle is too big
    for i in np.where(label == 2)[0]:
        if lvl[i] < 2:
            ind_filtered.append(ind[i])
            label_filtered.append(label[i])
        else:
            cnt += 1
    # Do not filter background
    for i in np.where(label == 1)[0]:
        ind_filtered.append(ind[i])
        label_filtered.append(label[i])

    print('Removed {} labels.'.format(cnt))

    return np.array(ind_filtered), np.array(label_filtered)

# Parameters
compute_sampling = False
compute_features = False

# APR file to segment
fpath_apr = r'/mnt/Data/wholebrain/multitile/000000/000000_000000/1_25x_tiling_file_t0_c1.apr'
fpath_labels = r'/mnt/Data/wholebrain/1_25x_tiling_file_t0_c1_Labels.npy'
# fpath_apr = r'/mnt/Data/Interneurons/output.apr'
# fpath_labels = r'/mnt/Data/Interneurons/manual_sparse_labels_membrane.npy'

# Instantiate APR and particle objects
labels = np.load(fpath_labels).squeeze().astype('uint16') # 0: not labeled - 1: cells - 2: background - 3: membrane
print('Labels loaded.')
# Read from APR file
apr = pyapr.APR()
parts = pyapr.ShortParticles()  # input particles can be float32 or uint16
pyapr.io.read(fpath_apr, apr, parts)
print('APR loaded.')
# Sample labels on APR grid
if compute_sampling:
    parts_train_C = pyapr.ShortParticles(apr.total_number_particles())
    parts_train_C = sample_labels_on_APR(labels, apr, parts_train_C)
    del labels
    print('Labels sampled on APR.')
    parts_train = np.array(parts_train_C, copy=False)
else:
    parts_train = np.load('parts_train.npy')

# Display sampled particles
# pyapr.viewer.parts_viewer(apr, parts_train_C)
if compute_features:
    t = time()
    # Compute gradient along a dimension (Sobel filter). dimension can be 0, 1 or 2
    # grad_x, grad_y, grad_z = compute_gradients(apr, parts)
    # print('Gradient computed.')

    gauss = gaussian_blur(apr, parts, sigma=1.5, size=11)
    print('Gaussian computed.')

    # Compute gradient magnitude (central finite differences)
    grad = compute_gradmag(apr, gauss)
    print('Gradient magnitude computed.')
    # Compute local standard deviation around each particle
    # local_std = compute_std(apr, parts, size=5)
    # print('STD computed.')
    # Compute lvl for each particle
    lvl = particle_levels(apr)
    print('Particle level computed.')
    # Compute difference of Gaussian
    dog = gaussian_blur(apr, parts, sigma=3, size=22) - gauss
    print('DOG computed.')
    lapl_of_gaussian = compute_laplacian(apr, gauss, )
    print('Laplacian of Gaussian computed.')

    print('Features computation took {} s.'.format(time()-t))

    # Aggregate filters in a feature array
    f = np.vstack((np.array(parts, copy=True),
                   lvl,
                   gauss,
                   grad,
                   lapl_of_gaussian,
                   dog
                   )).T
else:
    f = np.load('f.npy')

# from viewer.pyapr_napari import apr_to_napari_Image
# with napari.gui_qt():
#     viewer = napari.Viewer()
#     viewer.add_layer(apr_to_napari_Image(apr, dog))

f_names = ['Intensity',
               'lvl',
               'Gaussian',
               'grad of Gaussian',
               'laplacian of Gaussian',
               'difference of Gaussian'
           ]

# Fetch data that was manually labelled
ind_manual_label = (parts_train != 0)
y = parts_train[ind_manual_label]-1

# Remove label erased by APR conversion
ind_manual_label, y = filter_manual_labels(np.where(parts_train != 0)[0], y, f[ind_manual_label, 1])
x = f[ind_manual_label, :]

# Train random forest
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

clf = make_pipeline(preprocessing.StandardScaler(with_mean=True, with_std=True),
                    RandomForestClassifier(n_estimators=10, class_weight='balanced'))
t = time()
clf.fit(x, y)
print('Training took {} s.\n'.format(time()-t))

x_pred = clf.predict(x)

# Display training info
print('\n\n****** TRAINING RESULTS ******\n')
print('Total accuracy: {:0.2f}%'.format(np.sum(x_pred==y)/y.size*100))
print('Cell accuracy: {:0.2f}% ({} cell particles)'.format(np.sum((x_pred==y)*(y==0))/np.sum(y==0)*100, np.sum(y==0)))
print('Background accuracy: {:0.2f}% ({} background particles)'.format(np.sum((x_pred==y)*(y==1))/np.sum(y==1)*100, np.sum(y==1)))
print('Membrane accuracy: {:0.2f}% ({} membrane particles)'.format(np.sum((x_pred==y)*(y==2))/np.sum(y==2)*100, np.sum(y==2)))

# # Apply on whole dataset and display results
# # save_results = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/segmentation_results/labels.tif'
# # Predict particle type (cell, membrane or background) for each cell with the trained model
parts_pred = predict_on_APR_block(clf, f)

# Display inference info
print('\n\n****** INFERENCE RESULTS ******\n')
print('{} cell particles ({:0.2f}%)'.format(np.sum(parts_pred==0), np.sum(parts_pred==0)/len(parts_pred)*100))
print('{} background particles ({:0.2f}%)'.format(np.sum(parts_pred==1), np.sum(parts_pred==1)/len(parts_pred)*100))
print('{} membrane particles ({:0.2f}%)'.format(np.sum(parts_pred==2), np.sum(parts_pred==2)/len(parts_pred)*100))

# Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
parts_cells = (parts_pred == 0)

# Use opening to separate touching cells
pyapr.numerics.transform.opening(apr, parts_cells, binary=True, inplace=True)

# Apply connected component
cc = pyapr.LongParticles()
pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)

# Remove small and large objects
pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=4)
pyapr.numerics.transform.remove_large_objects(apr, cc, max_volume=256)

from viewer.pyapr_napari import display_segmentation
display_segmentation(apr, parts, cc)


#
# # Display results with napari
# from viewer.pyapr_napari import apr_to_napari_Image, apr_to_napari_Labels
# data = apr_to_napari_Image(apr, parts, name='Intensity image')
# lmap = apr_to_napari_Labels(apr, cc, name='Segmentation')
# with napari.gui_qt():
#     viewer = napari.Viewer(ndisplay=2)
#     # add the volume
#     viewer.add_layer(data)
#     # add labels
#     viewer.add_layer(lmap)
#
# # Save classifier
# # Note: this method is not perfect and it might break if scikit-learn version is not the same between the
# # dump and the loading. Use with caution (see https://scikit-learn.org/stable/modules/model_persistence.html)
# # from joblib import dump
# # dump(clf, '/media/sf_shared_folder_virtualbox/mouse_2P/data1/classifiers/random_forest_n100.joblib')
# # To load back use:
# # from joblib import load
# # toto = load('/media/sf_shared_folder_virtualbox/mouse_2P/data1/classifiers/random_forest_n100.joblib')
#
# # Get labeled objects patch coordinates (some id do not appear in cc)
# # minc, maxc = pyapr.numerics.transform.find_objects(apr, cc)
# # valid_labels = [(minc[i, :] < maxc[i, :]).all() for i in range(minc.shape[0])]
# # minc = minc[valid_labels, :]
# # maxc = maxc[valid_labels, :]
# #
# # # Compute the number of detected cells
# # print('Total number of detected cells: {}\n'.format(len(valid_labels)))
#
