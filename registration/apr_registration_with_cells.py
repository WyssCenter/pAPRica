"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pyapr
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import os
import napari
import cv2 as cv
from time import time
from joblib import load
import pandas as pd
from skimage.measure import regionprops_table, ransac
from skimage.transform import EuclideanTransform
from scipy.ndimage import fourier_shift

def display_registration(u1, u2, translation, scale=[1, 1, 1], contrast_limit=[0, 2000], opacity=0.7):
    """
    Use napari to display 2 registered volumes (works lazily with dask arrays).

    Parameters
    ----------
    u1: (array) volume 1
    u2: (array) volume 2
    translation: (array) translation vector (shape ndim) to position volume 2 with respect to volume 1
    scale: (array) scale vector (shape ndim) to display with the physical dimension if the sampling is anisotropic.
    contrast_limit: (array) contrast limit e.g. [0, 1000] to clip the displayed values and enhance the contrast.

    Returns
    -------
    None
    """

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(u1,
                         contrast_limits=contrast_limit,
                         multiscale=False,
                         name='1',
                         scale=scale,
                         opacity=opacity,
                         colormap='red',
                         blending='additive')
        viewer.add_image(u2,
                         contrast_limits=contrast_limit,
                         multiscale=False,
                         name='2',
                         translate=translation,
                         scale=scale,
                         opacity=opacity,
                         colormap='green',
                         blending='additive')


def display_segmentation(u, lmap, vdim=2, **kwargs):
    """
    Use napari to display volumetric intensity data with computed labels. Works with 2D (stack will be displayed
    like in ImageJ) or 3D rendering.

    Parameters
    ----------
    u: (array) volumetric intensity data
    lmap: (array) same shape as u, label map of segmented u
    vdim: (int) 2 or 3 depending on the desired rendering

    Returns
    -------
    None
    """
    if vdim not in (2, 3):
        raise ValueError('vdim not correct, expected 2 or 3 and got {}.'.format(vdim))

    if u.shape != lmap.shape:
        raise ValueError('Volumetric intensity array (shape {}) and '
                         'label map array (shape {}) have different shapes.'.format(u.shape, lmap.shape))

    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=vdim)
        # add the volume
        viewer.add_image(u, name='Intensity image', **kwargs)
        # add labels
        viewer.add_labels(lmap, name='segmentation', **kwargs)


def compare_segmentation(u, lmap1, lmap2, vdim=2, **kwargs):
    """
    Use napari to display volumetric intensity data with computed labels. Works with 2D (stack will be displayed
    like in ImageJ) or 3D rendering.

    Parameters
    ----------
    u: (array) volumetric intensity data
    lmap: (array) same shape as u, label map of segmented u
    vdim: (int) 2 or 3 depending on the desired rendering

    Returns
    -------
    None
    """
    if vdim not in (2, 3):
        raise ValueError('vdim not correct, expected 2 or 3 and got {}.'.format(vdim))

    if u.shape != lmap1.shape:
        raise ValueError('Volumetric intensity array (shape {}) and '
                         'label map 1 array (shape {}) have different shapes.'.format(u.shape, lmap1.shape))
    if u.shape != lmap2.shape:
        raise ValueError('Volumetric intensity array (shape {}) and '
                         'label map 2 array (shape {}) have different shapes.'.format(u.shape, lmap2.shape))

    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=vdim)
        # add the volumes
        viewer.add_image(u, name='Intensity image 1', **kwargs)
        viewer.add_image(u, name='Intensity image 2', translate=[0, 0, u.shape[-1]], **kwargs)
        # add labels
        viewer.add_labels(lmap1, name='Segmentation 1', **kwargs)
        viewer.add_labels(lmap2, name='Segmentation 2', translate=[0, 0, u.shape[-1]], **kwargs)


def get_projection(c):
    """
    Project 3D cell positions in on (x,y), (x,z) and (y,z).

    Parameters
    ----------
    c: (2D array) cell position with shape (n_cell, 3)

    Returns
    -------
    The 3 projections on (x,y), (x,z) and (y,z).
    """

    c_xy = np.vstack((c[:, 0], c[:, 1])).T
    c_xz = np.vstack((c[:, 0], c[:, 2])).T
    c_yz = np.vstack((c[:, 1], c[:, 2])).T
    return (c_xy, c_xz, c_yz)


def find_rigid_ransac(c1, c2, plot=False, verbose=True):
    """
    Find rigid registration based on RANSAC.

    Parameters
    ----------
    c1: (2D array) List of coordinates for the first volume. Shape should be [n, 3]
    c2: (2D array) List of coordinates for the second volume. Shape should be [n, 3]
    plot: (bool) Control result display
    verbose: (bool) Control result printing in terminal

    Returns
    -------
    d_mes an array containing the three computed shifts [dx, dy, dz] for the rigid registration.
    """

    if c1.shape != c2.shape:
        raise TypeError('c1 and c2 shapes are different. Hint: use Flann robust matching before this function.'.format(c1.shape[1], c2.shape[1]))

    if c1.shape[1] != 3:
        raise TypeError('c1 and c2 are dim {}, expected dim 3.'.format(c1.shape[1], c2.shape[1]))


    # Project on each axis for estimating translation
    c1_proj = get_projection(c1)
    c2_proj = get_projection(c2)

    # Robustly find translation parameter using ransac
    dx = []
    dy = []
    dz = []
    for ii, src, dst in zip(range(len(c1_proj)), c1_proj, c2_proj):
        model_robust, inliers = ransac((src, dst), EuclideanTransform, min_samples=int(len(src)*0.7),
                                       residual_threshold=2, max_trials=1000)
        src_reg = model_robust(src)
        if plot:
            fig, ax = plt.subplots(1, 2, sharex =True, sharey=True)
            ax[0].plot(src[:, 0], src[:, 1], 'k+', label='src')
            ax[0].plot(dst[:, 0], dst[:, 1], 'r+', label='dst')
            ax[0].plot(src_reg[:, 0], src_reg[:, 1], 'ro', label='src_reg')
            ax[0].set_title('Projection {}'.format(ii+1))
            ax[1].plot(src[inliers, 0], src[inliers, 1], 'k+', label='src')
            ax[1].plot(dst[inliers, 0], dst[inliers, 1], 'r+', label='dst')
            ax[1].set_title('Projection {} only inliers'.format(ii+1))
            ax[1].plot(src_reg[inliers, 0], src_reg[inliers, 1], 'ro', label='src_reg')
            plt.legend()

        if verbose:
            print('{:0.2f}% inliers were found'.format(np.sum(inliers) / inliers.size * 100))

        if ii == 0:
            dx.append(model_robust.translation[0])
            dy.append(model_robust.translation[1])
        elif ii == 1:
            dx.append(model_robust.translation[0])
            dz.append(model_robust.translation[1])
        elif ii == 2:
            dy.append(model_robust.translation[0])
            dz.append(model_robust.translation[1])

    dx = np.array(dx).mean()
    dy = np.array(dy).mean()
    dz = np.array(dz).mean()
    d_mes = np.array([dx, dy, dz])

    # Display registered subset
    if plot:
        d_mes_mat = np.tile(d_mes, (c1.shape[0], 1))
        c1_reg = c1 + d_mes_mat
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(c1_reg[:, 0], c1_reg[:, 1], c1_reg[:, 2], color='k', label='ref registered to shuffled')
        ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2], color='r', label='shuffled + translation')
        plt.legend()

    return d_mes


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


def predict_on_APR(clf, x):
    # Predict on numpy array
    t = time()
    y_pred = clf.predict(x)
    print('Prediction took {} s.\n'.format(time()-t))

    # Transform numpy array to ParticleData
    parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

    return parts_pred


def segment_apr(apr, parts, clf):
    # Compute features
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
    print('Features computation took {} s.'.format(time() - t))

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
    f_names = ['Intensity',
               'lvl',
               'grad_x',
               'grad_y',
               'grad_z',
               'grad_mag',
               'local_std',
               'dog'
               ]

    # plt.figure()
    # plt.imshow(np.corrcoef(f.T), cmap='jet')
    # plt.colorbar()
    # plt.xticks(np.arange(len(f_names)), f_names, rotation=45)
    # plt.yticks(np.arange(len(f_names)), f_names)
    # plt.tight_layout()

    # Apply on whole dataset and display results
    # Predict particle type (cell, membrane or background) for each cell with the trained model
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
    return cc


def match_features_flann(f1, f2, c1, c2, lowe_ratio=0.7):

    # Test for correct inputs
    if c1.shape[1] != c2.shape[1]:
        raise TypeError('c1 is dim {} while c2 is dim {}, expected the same value.'.format(c1.shape[1], c2.shape[1]))

    if lowe_ratio<0 or lowe_ratio>1:
        raise ValueError('Lowe ratio is {}, expected between 0 and 1.'.format(lowe_ratio))

    ndim = c1.shape[1]
    # Match cells descriptors by using Flann method
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.float32(f1), np.float32(f2), k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < lowe_ratio*n.distance:
            good.append(m)

    # Reorder cells with found matches
    c1 = np.float64([c1[m.queryIdx] for m in good]).reshape(-1, ndim)
    c2 = np.float64([c2[m.trainIdx] for m in good]).reshape(-1, ndim)

    return (c1, c2)


def get_point_cloud_distance(c1, c2, lowe_ratio=0.3):
    """
    Obtain the mean distance between point cloud (should work in any dimension).
    Return the percentage of matched point with Flann method and evaluate the
    mean distance only for matched point.

    Parameters
    ----------
    c1: (2D array) coordinate of points in 1st cloud with shape (n_points, dim)
    c2: (2D array) coordinate of points in 1st cloud with shape (n_points, dim)
    lowe_ratio: (float) ratio between first and second nearest neighbors to consider a
                reliable match. Must be between in ]0, 1[.

    Returns
    -------
    match_percentage: (float) percentage of matched points
    distance_3D: (float) average distance for matched points

    """

    if lowe_ratio<0 or lowe_ratio>1:
        raise ValueError('Lowe ratio is {}, expected between 0 and 1.'.format(lowe_ratio))

    # Match cells by using Flann method
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.float32(c1), np.float32(c2), k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < lowe_ratio * n.distance:
            good.append(m)

    match_percentage = len(good) / c1.shape[0]

    # Reorder cells with found matches
    c1 = np.float64([c1[m.queryIdx] for m in good]).reshape(-1, 3)
    c2 = np.float64([c2[m.trainIdx] for m in good]).reshape(-1, 3)

    # Compute 3D distance
    distance_3D = np.linalg.norm(c1 - c2, axis=1)
    distance_3D_mean = np.mean(distance_3D)
    distance_3D_median = np.median(distance_3D)

    return match_percentage, distance_3D_mean, distance_3D_median


def get_patch(apr, parts, minc, maxc, display=False):
    image_patch = np.array(pyapr.numerics.reconstruction.recon_patch(apr, parts,
                                                            minc[0], maxc[0],
                                                            minc[1], maxc[1],
                                                            minc[2], maxc[2]))
    if display:
        with napari.gui_qt():
            viewer = napari.Viewer(ndisplay=2)
            # add the volume
            viewer.add_image(image_patch, name='Patch')

    return image_patch

PROPS = {
    'Area': 'area',
    # 'BoundingBox': 'bbox',
    # 'BoundingBoxArea': 'bbox_area',
    # 'CentralMoments': 'moments_central',
    # 'Centroid': 'centroid',
    # 'ConvexArea': 'convex_area',
    # 'ConvexHull',
    # 'ConvexImage': 'convex_image',
    # 'Coordinates': 'coords',
    # 'Eccentricity': 'eccentricity',
    'EquivDiameter': 'equivalent_diameter',
    'EulerNumber': 'euler_number',
    # 'Extent': 'extent',
    # 'Extrema',
    # 'FeretDiameterMax': 'feret_diameter_max',
    # 'FilledArea': 'filled_area',
    # 'FilledImage': 'filled_image',
    # 'HuMoments': 'moments_hu',
    # 'Image': 'image',
    # 'InertiaTensor': 'inertia_tensor',
    'InertiaTensorEigvals': 'inertia_tensor_eigvals',
    # 'IntensityImage': 'intensity_image',
    # 'Label': 'label',
    # 'LocalCentroid': 'local_centroid',
    'MajorAxisLength': 'major_axis_length',
    'MaxIntensity': 'max_intensity',
    'MeanIntensity': 'mean_intensity',
    'MinIntensity': 'min_intensity',
    'MinorAxisLength': 'minor_axis_length',
    # 'Moments': 'moments',
    # 'NormalizedMoments': 'moments_normalized',
    # 'Orientation': 'orientation',
    # 'Perimeter': 'perimeter',
    # 'CroftonPerimeter': 'perimeter_crofton',
    # 'PixelIdxList',
    # 'PixelList',
    # 'Slice': 'slice',
    # 'Solidity': 'solidity',
    # 'SubarrayIdx'
    # 'WeightedCentralMoments': 'weighted_moments_central',
    'WeightedCentroid': 'weighted_centroid',
    # 'WeightedHuMoments': 'weighted_moments_hu',
    # 'WeightedLocalCentroid': 'weighted_local_centroid',
    # 'WeightedMoments': 'weighted_moments',
    # 'WeightedNormalizedMoments': 'weighted_moments_normalized'
}


def get_features_apr(apr, parts, cc):
    # Get labeled objects patch coordinates (some id do not appear in cc)
    minc, maxc = pyapr.numerics.transform.find_objects(apr, cc)
    valid_labels = [(minc[i, :] < maxc[i, :]).all() for i in range(minc.shape[0])]
    minc = minc[valid_labels, :]
    maxc = maxc[valid_labels, :]

    # Compute the number of detected cells
    print('Total number of detected cells: {}\n'.format(len(valid_labels)))

    # Loop to reconstruct object and compute properties on them
    mask = cc>0
    n_object = minc.shape[0]
    for i in range(n_object):
        image_patch = get_patch(apr, parts, minc[i, :], maxc[i, :])
        label_patch = get_patch(apr, mask, minc[i, :], maxc[i, :])
        l = regionprops_table(label_image=label_patch, intensity_image=image_patch, properties=PROPS.values())
        for num in range(3):
            l['centroid-' + str(num)] = l['centroid-' + str(num)] + minc[i, num]
            l['weighted_centroid-' + str(num)] = l['weighted_centroid-' + str(num)] + minc[i, num]
            l['local_centroid-' + str(num)] = l['local_centroid-' + str(num)] + minc[i, num]
            l['bbox-' + str(num)] = minc[i, num]
            l['bbox-' + str(num + 3)] = maxc[i, num]
        if i == 0:
            cells = pd.DataFrame(l)
        else:
            cells.append(l, ignore_index=True)
    return cells


def find_rigid_ransac(c1, c2, plot=False, verbose=True):
    """
    Find rigid registration based on RANSAC.

    Parameters
    ----------
    c1: (2D array) List of coordinates for the first volume. Shape should be [n, 3]
    c2: (2D array) List of coordinates for the second volume. Shape should be [n, 3]
    plot: (bool) Control result display
    verbose: (bool) Control result printing in terminal

    Returns
    -------
    d_mes an array containing the three computed shifts [dx, dy, dz] for the rigid registration.
    """

    if c1.shape != c2.shape:
        raise TypeError('c1 and c2 shapes are different. Hint: use Flann robust matching before this function.')

    if c1.shape[1] != 3:
        raise TypeError('c1 and c2 are dim {}, expected dim 3.'.format(c1.shape[1], c2.shape[1]))


    # Project on each axis for estimating translation
    c1_proj = get_projection(c1)
    c2_proj = get_projection(c2)

    # Robustly find translation parameter using ransac
    dx = []
    dy = []
    dz = []
    for ii, src, dst in zip(range(len(c1_proj)), c1_proj, c2_proj):
        model_robust, inliers = ransac((src, dst), EuclideanTransform, min_samples=3,
                                       residual_threshold=2, max_trials=100)
        src_reg = model_robust(src)
        if plot:
            fig, ax = plt.subplots(1, 2, sharex =True, sharey=True)
            ax[0].plot(src[:, 0], src[:, 1], 'k+', label='src')
            ax[0].plot(dst[:, 0], dst[:, 1], 'r+', label='dst')
            ax[0].plot(src_reg[:, 0], src_reg[:, 1], 'ro', label='src_reg')
            ax[0].set_title('Projection {}'.format(ii+1))
            ax[1].plot(src[inliers, 0], src[inliers, 1], 'k+', label='src')
            ax[1].plot(dst[inliers, 0], dst[inliers, 1], 'r+', label='dst')
            ax[1].set_title('Projection {} only inliers'.format(ii+1))
            ax[1].plot(src_reg[inliers, 0], src_reg[inliers, 1], 'ro', label='src_reg')
            plt.legend()

        if verbose:
            print('{:0.2f}% inliers were found'.format(np.sum(inliers) / inliers.size * 100))

        if ii == 0:
            dx.append(model_robust.translation[0])
            dy.append(model_robust.translation[1])
        elif ii == 1:
            dx.append(model_robust.translation[0])
            dz.append(model_robust.translation[1])
        elif ii == 2:
            dy.append(model_robust.translation[0])
            dz.append(model_robust.translation[1])

    dx = np.array(dx).mean()
    dy = np.array(dy).mean()
    dz = np.array(dz).mean()
    d_mes = np.array([dx, dy, dz])

    # Display registered subset
    if plot:
        d_mes_mat = np.tile(d_mes, (c1.shape[0], 1))
        c1_reg = c1 + d_mes_mat
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(c1_reg[:, 0], c1_reg[:, 1], c1_reg[:, 2], color='k', label='ref registered to shuffled')
        ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2], color='r', label='shuffled + translation')
        plt.legend()

    return d_mes



# Parameters
folder_path = r'/media/sf_shared_folder_virtualbox/PV_interneurons'
image_path = r'substack.tif'
clf_path = r'classifiers/random_forest_n100.joblib'
overlap_size = 512
overlap_shift = 1024


# Load data
image_1 = imread(os.path.join(folder_path, image_path))
clf = load(os.path.join(folder_path, clf_path))

# Resize image with overlap size
image_1 = image_1[:, :, overlap_shift:overlap_shift+overlap_size]
print('Overlap data size: {:.2f} Mo.'.format(image_1.size*2/1e6))
# image_ini = ((image_ini/4)+100).astype('uint16') # The classifier was train with this intensity transform

# Apply a random shift for image_2
np.random.seed(0)
random_displacement = np.random.randn(3)*[5, 10, 10]
image_2 = np.abs(np.fft.ifftn(fourier_shift(np.fft.fftn(image_1), shift=random_displacement)))
image_2 = (image_2 + np.random.randn(*image_2.shape)*100)
image_2[image_2 < 0] = 0
image_2[image_2 > 2**16-1] = 2**16-1
image_2 = image_2.astype('uint16')

# Visualize both volumes using Napari with the compensated shift
# display_registration(image_1, image_2, contrast_limit=[0, 1000], translation=[0, 0, 0])
# display_registration(image_1, image_2, contrast_limit=[0, 1000], translation=-random_displacement)

# Convert volumes to APR
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working
# Parameters for Nissl
# par.sigma_th = 26.0
# par.grad_th = 3.0
# par.Ip_th = 253.0
# par.rel_error = 0.2
# par.gradient_smoothing = 2
# Parameters for PV_interneurons
par.sigma_th = 562.0
par.grad_th = 49.0
par.Ip_th = 903.0
par.rel_error = 0.2
par.gradient_smoothing = 2.0
par.dx = 1
par.dy = 1
par.dz = 3
apr_1, parts_1 = pyapr.converter.get_apr(image_1, verbose=True, params=par)
apr_2, parts_2 = pyapr.converter.get_apr(image_2, verbose=True, params=par)

# Segment particles on both overlapping volumes
t = time()
cc_1 = segment_apr(apr_1, parts_1, clf)
print('Segmentation of 1st APR elapsed time: {:.3f} s.'.format(time()-t))
t = time()
cc_2 = segment_apr(apr_2, parts_2, clf)
print('Segmentation of 1st APR elapsed time: {:.3f} s.'.format(time()-t))

# Display segmentations
lmap_1 = np.array(pyapr.numerics.reconstruction.recon_pc(apr_1, cc_1))
lmap_2 = np.array(pyapr.numerics.reconstruction.recon_pc(apr_2, cc_2))
# display_registration(lmap_1, lmap_2, translation=-random_displacement)

# Compute features on both segmented volumes
f_1 = pd.DataFrame(regionprops_table(label_image=lmap_1, intensity_image=image_1, properties=PROPS.values()))
f_2 = pd.DataFrame(regionprops_table(label_image=lmap_2, intensity_image=image_2, properties=PROPS.values()))

coord = 'weighted_centroid-'

# TODO: display segmentation and centroid in napari
c_1 = f_1[[coord + str(i) for i in range(3)]].to_numpy()
f_1_numpy = f_1.drop([coord + str(i) for i in range(3)], axis='columns').to_numpy()

c_2 = f_2[[coord + str(i) for i in range(3)]].to_numpy()
f_2_numpy = f_2.drop([coord + str(i) for i in range(3)], axis='columns').to_numpy()

t = time()
c1, c2 = match_features_flann(f_1_numpy, f_2_numpy, c_1, c_2, lowe_ratio=0.7)
print('Flann match elapsed time: {:.3f} s.'.format(time()-t))

# def display_registration_points(u1, u2, c1, c2, translation, scale=[1, 1, 1], contrast_limit=[0, 30000], opacity=0.7):
#     with napari.gui_qt():
#         viewer = napari.Viewer()
#         viewer.add_image(u1,
#                          contrast_limits=contrast_limit,
#                          multiscale=False,
#                          name='1',
#                          scale=scale,
#                          opacity=opacity,
#                          colormap='red',
#                          blending='additive')
#         viewer.add_points(c1)
#         viewer.add_image(u2,
#                          contrast_limits=contrast_limit,
#                          multiscale=False,
#                          name='2',
#                          translate=translation,
#                          scale=scale,
#                          opacity=opacity,
#                          colormap='green',
#                          blending='additive')
#         viewer.add_points(c2)


# Display in 3D c1 and c2_reg before computing the transform with ransac
# c2_reg = c2 - np.tile(random_displacement, (c2.shape[0], 1))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], color='g', label='image_1', alpha=0.5)
# ax.scatter(c2_reg[:, 0], c2_reg[:, 1], c2_reg[:, 2], color='r', label='image_2 reg', alpha=0.5)
# plt.legend()
#
# plt.figure()
# plt.scatter(c1[:, 0], c1[:, 1], color='g', label='image_1', alpha=0.5)
# plt.scatter(c2_reg[:, 0], c2_reg[:, 1], color='r', label='image_2 reg', alpha=0.5)
# plt.xlabel('0')
# plt.ylabel('1')
# plt.legend()
#
# plt.figure()
# plt.scatter(c1[:, 0], c1[:, 2], color='g', label='image_1', alpha=0.5)
# plt.scatter(c2_reg[:, 0], c2_reg[:, 2], color='r', label='image_2 reg', alpha=0.5)
# plt.xlabel('0')
# plt.ylabel('2')
# plt.legend()
#
# plt.figure()
# plt.scatter(c1[:, 1], c1[:, 2], color='g', label='image_1', alpha=0.5)
# plt.scatter(c2_reg[:, 1], c2_reg[:, 2], color='r', label='image_2 reg', alpha=0.5)
# plt.xlabel('1')
# plt.ylabel('2')
# plt.legend()

t = time()
d_mes = find_rigid_ransac(c1, c2, plot=False, verbose=True)
print('RANSAC elapsed time: {:.3f} s.'.format(time()-t))

accuracy = (d_mes-random_displacement)
for i, ax in enumerate(['z', 'x', 'y']):
    print('Registration error for {} axis: {:0.3f} pixel.'.format(ax, accuracy[i]))