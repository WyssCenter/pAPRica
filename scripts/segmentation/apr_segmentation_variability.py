"""
This script is to study the difference in segmentation when we vary APR parameters
In this script I perform cell clustering on nissl data from 2P. First the data is converted to APR
then segmented and clustered.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pyapr
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import napari
from time import time
from joblib import load
import pandas as pd
from skimage.measure import regionprops, label
import cv2 as cv

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


def dice_coeff(u1, u2, plot=False, ax=None):

    """
    This function computes the Dice coefficients for 2D binary masks.

    Parameters
    ----------
    u1: (bool) 1st 2D binary mask
    u2: (bool) 2nd 2D binary mask
    plot: (bool) control results display
    ax: (ax object) display results in given ax object.

    Returns
    -------
    Dice index for foreground and background
    """

    dice_fg = np.sum((u1*u2)==1)/((np.sum(u1==1)+np.sum(u2==1))/2)
    dice_bg = np.sum((u1==0)*(u2==0))/((np.sum(u1==0)+np.sum(u2==0))/2)

    if plot:
        R = u1.astype('uint8')*255
        B = u2.astype('uint8')*255
        G = np.zeros_like(u1)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.imshow(np.dstack((R, G, B)))
        ax.set_axis_off()
        ax.set_title('Segmentation mask comparison')
    return [dice_fg, dice_bg]


def get_centroids_from_smap(smap):
    """
    Return cells centroid from probability map.

    Parameters
    ----------
    pmap: (3D array) segmentation map of segmented cells

    Returns
    -------
    Cell centroids
    """

    cells = []
    for cell in regionprops(label(smap)):
        cells.append(cell['centroid'])
    return np.array(cells)


def get_point_cloud_distance(c1, c2):
    """
    Obtain the mean distance between point cloud (should work in any dimension).
    Return the percentage of matched point with Flann method and evaluate the
    mean distance only for matched point.

    Parameters
    ----------
    c1: (2D array) coordinate of points in 1st cloud with shape (n_points, dim)
    c2: (2D array) coordinate of points in 1st cloud with shape (n_points, dim)

    Returns
    -------
    match_percentage: (float) percentage of matched points
    distance_3D: (float) average distance for matched points

    """

    # Match cells by using Flann method
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.float32(c1), np.float32(c2), k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
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


def compare_smap(smap1, smap2, verbose=True):
    dice = []
    # Compute Dice and PSNR for each frame
    for n in range(smap1.shape[0]):
        dice.append(dice_coeff(smap1[n], smap2[n]))
    dice = np.array(dice)

    # Compare centroid position
    c1 = get_centroids_from_smap(smap1)
    c2 = get_centroids_from_smap(smap2)
    centroid_match_percentage, distance_3D_mean, distance_3D_median = get_point_cloud_distance(c1, c2)

    if verbose:
        print('Average 3D distance is {:.3f} pixel.'.format(distance_3D_mean))
        print('Median 3D distance is {:.3f} pixel.'.format(distance_3D_median))
        print('Percentage of matched point is {:.3f}.'.format(centroid_match_percentage*100))
        print('Dice index is {:.3f} +- {:.3f}.'.format(dice.mean(), dice.std()))

    results = {'dice_mean': dice.mean(),
               'dice_std': dice.std(),
               'centroid_3D_distance_mean': distance_3D_mean,
               'centroid_3D_distance_median': distance_3D_median,
               'centroid_match_percentage': centroid_match_percentage}

    return results


def display_segmentations(u, lmap, vdim=2):
    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=vdim)
        # add the volume
        viewer.add_image(u, name='Intensity image')
        # add labels
        for i, l in enumerate(lmap):
            l[l>0] = i+1
            viewer.add_labels(l, name='segmentation {}'.format(i))

# Parameters
folder_path = r'/media/sf_shared_folder_virtualbox/PV_interneurons'
image_path = r'substack.tif'
clf_path = r'/media/sf_shared_folder_virtualbox/PV_interneurons/classifiers/random_forest_n100.joblib'
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working
par.sigma_th = 562.0
par.grad_th = 49.0
par.Ip_th = 903.0
par.rel_error = 0.2
par.gradient_smoothing = 2.0
par.dx = 1
par.dy = 1
par.dz = 3
# Load image and transform it to APR
data = imread(os.path.join(folder_path, image_path))
data = data[:, :, 800:800+512]
apr_ref, parts_ref = pyapr.converter.get_apr(data, verbose=True, params=par)
print('CR: {:.2f}'.format(data.size/apr_ref.total_number_particles()))

# Load classifier
clf = load(clf_path)

# Perform reference segmentation segmentation
cc_ref = segment_apr(apr_ref, parts_ref, clf)
smap_ref = cc_ref > 0
smap_ref_pxl = np.array(pyapr.numerics.reconstruction.recon_pc(apr_ref, smap_ref))

results = pd.DataFrame(columns=['dice_mean',
                                'dice_std', 'centroid_3D_distance_mean',
                                'centroid_3D_distance_median', 'centroid_match_percentage'])
for rel_error in [0.1, 0.2, 0.4, 0.8]:
    par.rel_error = rel_error
    apr, parts = pyapr.converter.get_apr(data, verbose=True, params=par)
    cc = segment_apr(apr, parts, clf)
    smap = cc > 0
    smap_pxl = np.array(pyapr.numerics.reconstruction.recon_pc(apr, smap))
    res = compare_smap(smap_ref_pxl, smap_pxl)
    results = results.append(res, ignore_index=True)
results['rel_error'] = [0.1, 0.2, 0.4, 0.8]



display_segmentations(data, [smap_ref_pxl, smap_pxl])