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


def compute_features(apr, parts):
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
    lapl_of_gaussian = compute_laplacian(apr, gauss)
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

    return f


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


# APR file to segment
fpath_apr = r'/mnt/Data/Interneurons/output.apr'
fpath_labels = r'/mnt/Data/Interneurons/manual_sparse_labels_membrane.npy'

# Instantiate APR and particle objects
apr = pyapr.APR()
parts = pyapr.ShortParticles()  # input particles can be float32 or uint16
labels = np.load(fpath_labels).squeeze().astype('uint16') # 0: not labeled - 1: cells - 2: background - 3: membrane
print('Labels loaded.')
# Read from APR file
pyapr.io.read(fpath_apr, apr, parts)
print('APR loaded.')
# Sample labels on APR grid
parts_train_C = pyapr.ShortParticles(apr.total_number_particles())
parts_train_C = sample_labels_on_APR(labels, apr, parts_train_C)
del labels
print('Labels sampled on APR.')
parts_train = np.array(parts_train_C, copy=False)

t = time()
# Compute gradient along a dimension (Sobel filter). dimension can be 0, 1 or 2
grad_x, grad_y, grad_z = compute_gradients(apr, parts)
print('g')

f = compute_features(apr, parts)

# plt.figure()
# print('k')
# plt.imshow(np.corrcoef(f.T), cmap='jet')
# plt.colorbar()
# plt.xticks(np.arange(len(f_names)), f_names, rotation=45)
# plt.yticks(np.arange(len(f_names)), f_names)
# plt.tight_layout()
# print('1')

# Fetch data that was manually labelled
ind_manual_label = (parts_train != 0)
print('2')
x = f[ind_manual_label, :]
print('3')
y = parts_train[ind_manual_label]-1
print('4')

# Train SVM classification
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

print('5')
clf = make_pipeline(preprocessing.StandardScaler(with_mean=True, with_std=True),
                    RandomForestClassifier(n_estimators=100, class_weight='balanced'))
print('6')
t = time()
clf.fit(x, y)
print('Training took {} s.\n'.format(time()-t))

x_pred = clf.predict(x)

print('\n\n****** TRAINING RESULTS ******\n')
print('Total accuracy: {:0.2f}%'.format(np.sum(x_pred==y)/y.size*100))
print('Cell accuracy: {:0.2f}% ({} cell particles)'.format(np.sum((x_pred==y)*(y==0))/np.sum(y==0)*100, np.sum(y==0)))
print('Background accuracy: {:0.2f}% ({} background particles)'.format(np.sum((x_pred==y)*(y==1))/np.sum(y==1)*100, np.sum(y==1)))
print('Membrane accuracy: {:0.2f}% ({} membrane particles)'.format(np.sum((x_pred==y)*(y==2))/np.sum(y==2)*100, np.sum(y==2)))

# Display tree
# from sklearn import tree
# tree.plot_tree(clf[1].estimators_[0],
#                feature_names=f_names,
#                class_names=['Cells', 'Membranes', 'Background'],
#                filled=True)
#
# # Apply on whole dataset and display results
# # save_results = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/segmentation_results/labels.tif'
# # Predict particle type (cell, membrane or background) for each cell with the trained model
parts_pred = predict_on_APR_block(clf, f)
print('7')
# Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
parts_cells = (parts_pred == 0)
print('8')
# Remove small holes to get the misclassified nuclei
parts_cells = pyapr.numerics.transform.remove_small_holes(apr, parts_cells, min_volume=500)
print('9')
# Opening to better separate touching cells
pyapr.numerics.transform.opening(apr, parts_cells, radius=1, inplace=True)
print('10')
# Apply connected component
cc = pyapr.ShortParticles()
print('11')
pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)
print('12')
# Remove small objects
pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=200)
print('13')

# Display results with napari
from pipapr.viewer import apr_to_napari_Image, apr_to_napari_Labels
data = apr_to_napari_Image(apr, parts, name='Intensity image')
lmap = apr_to_napari_Labels(apr, cc, name='Segmentation')
with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=2)
    # add the volume
    viewer.add_layer(data)
    # add labels
    viewer.add_layer(lmap)

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

