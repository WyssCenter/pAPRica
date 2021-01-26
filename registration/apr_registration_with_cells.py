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
                         name='Left',
                         scale=scale,
                         opacity=opacity,
                         colormap='red',
                         blending='additive')
        viewer.add_image(u2,
                         contrast_limits=contrast_limit,
                         multiscale=False,
                         name='Right',
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
        viewer.add_image(u, name='Intensity image left', **kwargs)
        viewer.add_image(u, name='Intensity image right', translate=[0, 0, u.shape[-1]], **kwargs)
        # add labels
        viewer.add_labels(lmap1, name='Segmentation left', **kwargs)
        viewer.add_labels(lmap2, name='Segmentation right', translate=[0, 0, u.shape[-1]], **kwargs)


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
path = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/2P_FF0.7_AF3_Int50_Zoom3_Stacks.tif'
overlap = 25

# Read image and divide it in two with overlap %
image_ini = imread(path)
image_ini = ((image_ini/4)+100).astype('uint16') # The classifier was train with this intensity transform
s_ini = image_ini.shape
# Image left and image right are overlapping region where we are looking for the
# correct registration parameters
image_left = image_ini[:, :, :int(s_ini[2]/2*(1+overlap/100))]

# Apply a random shift for image_right
np.random.seed(0)
random_displacement = np.random.randn(3)*[1, 5, 5]
image_ini_shifted = np.abs(np.fft.ifftn(fourier_shift(np.fft.fftn(image_ini), shift=random_displacement))).astype('uint16')
image_right = image_ini_shifted[:, :, int(s_ini[2]/2*(1-overlap/100)):]

# Visualize both volumes using Napari with the compensated shift
display_registration(image_left, image_right, contrast_limit=[0, 40000],
                     translation=[0, 0, int(s_ini[2]/2*(1-overlap/100))])
display_registration(image_left, image_right, contrast_limit=[0, 40000],
                     translation=[0, 0, int(s_ini[2]/2*(1-overlap/100))]-random_displacement)

# Convert volumes to APR
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working
par.sigma_th = 26.0
par.grad_th = 3.0
par.Ip_th = 253.0
par.rel_error = 0.2
par.gradient_smoothing = 2
apr_left, parts_left = pyapr.converter.get_apr(image_left, verbose=True, params=par)
apr_right, parts_right = pyapr.converter.get_apr(image_right, verbose=True, params=par)

# Segment particles on both overlapping volumes
clf = load('/media/sf_shared_folder_virtualbox/mouse_2P/data1/classifiers/random_forest_n100.joblib')
cc_left = segment_apr(apr_left, parts_left, clf)
cc_right = segment_apr(apr_right, parts_right, clf)

# Compute features on both segmented volumes
lmap_left = np.array(pyapr.numerics.reconstruction.recon_pc(apr_left, cc_left))
lmap_right = np.array(pyapr.numerics.reconstruction.recon_pc(apr_right, cc_right))
display_registration(lmap_left, lmap_right, translation=[0, 0, int(s_ini[2]/2*(1-overlap/100))]-random_displacement)
f_left = pd.DataFrame(regionprops_table(label_image=lmap_left, intensity_image=image_left, properties=PROPS.values()))
f_right = pd.DataFrame(regionprops_table(label_image=lmap_right, intensity_image=image_right, properties=PROPS.values()))

coord = 'weighted_centroid-'

# TODO: check the conditions on ind to include all the overlapping cells
c_left = f_left[[coord + str(i) for i in range(3)]]
# f_left.drop([coord + str(i) for i in range(3)], axis='columns', inplace=True)
ind = c_left[coord + str(2)] > s_ini[2]/2 - overlap
f_left_overlap = f_left[ind].dropna(axis='columns')
c_left = c_left[ind].to_numpy()
c_left[:, 2] = c_left[:, 2] - s_ini[2]/2*(1-overlap/100)

c_right = f_right[[coord + str(i) for i in range(3)]]
# f_right.drop([coord + str(i) for i in range(3)], axis='columns', inplace=True)
ind = c_right[coord + str(2)] < s_ini[2]/2*(overlap/100) + overlap
f_right_overlap = f_right[ind].dropna(axis='columns')
c_right = c_right[ind].to_numpy()

c1, c2 = match_features_flann(f_left_overlap, f_right_overlap, c_left, c_right, lowe_ratio=0.7)

def display_registration_points(u1, u2, c1, c2, translation, scale=[1, 1, 1], contrast_limit=[0, 30000], opacity=0.7):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(u1,
                         contrast_limits=contrast_limit,
                         multiscale=False,
                         name='Left',
                         scale=scale,
                         opacity=opacity,
                         colormap='red',
                         blending='additive')
        viewer.add_points(c1)
        viewer.add_image(u2,
                         contrast_limits=contrast_limit,
                         multiscale=False,
                         name='Right',
                         translate=translation,
                         scale=scale,
                         opacity=opacity,
                         colormap='green',
                         blending='additive')
        viewer.add_points(c2)


# Display in 3D c1 and c2_reg before computing the transform with ransac
c2_reg = c2 - np.tile(random_displacement, (c2.shape[0], 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], color='k', label='image_left')
ax.scatter(c2_reg[:, 0], c2_reg[:, 1], c2_reg[:, 2], color='r', label='image_right reg')
plt.legend()

plt.figure()
plt.scatter(c1[:, 0], c1[:, 1], color='k', label='image_left')
plt.scatter(c2_reg[:, 0], c2_reg[:, 1], color='r', label='image_right reg')
plt.xlabel('0')
plt.ylabel('1')
plt.legend()

plt.figure()
plt.scatter(c1[:, 0], c1[:, 2], color='k', label='image_left')
plt.scatter(c2_reg[:, 0], c2_reg[:, 2], color='r', label='image_right reg')
plt.xlabel('0')
plt.ylabel('2')
plt.legend()

plt.figure()
plt.scatter(c1[:, 1], c1[:, 2], color='k', label='image_left')
plt.scatter(c2_reg[:, 1], c2_reg[:, 2], color='r', label='image_right reg')
plt.xlabel('1')
plt.ylabel('2')
plt.legend()

d_mes = find_rigid_ransac(c1, c2, plot=True, verbose=True)

# Keep only cells on the overlapping area
# l = s_ini[2]
# centroids = f_left['centroid-2']
# ind_left = centroids > l*(1-overlap/100)


#
#
#
#
# # Create sub-images on the overlapping area
# parts_left = np.array(parts_left)
# parts_right = np.array(parts_right)
#
# # Get particle position and level for left image
# apr_it = apr_left.iterator()
# part_position_left = []
# part_level_left = []
# part_intensity_left = []
# for level in range(apr_it.level_min(), apr_it.level_max()+1):
#     for z in range(apr_it.z_num(level)):
#         for x in range(apr_it.x_num(level)):
#             for idx in range(apr_it.begin(level, z, x), apr_it.end()):
#                 y = apr_it.y(idx)
#                 part_position_left.append([z, y, x])
#                 part_level_left.append(level)
#                 part_intensity_left.append(parts_left[idx])
#
# # Find the Nth particles with highest brightness
# N = 1000
# part_intensity_left = np.array(part_intensity_left)
# ind_max = np.argpartition(part_intensity_left, kth=len(part_intensity_left)-N)
# ind_max = ind_max[-N:]
# subpart_position_left = np.array(part_position_left)
# subpart_position_left = subpart_position_left[ind_max, :]
# subpart_level_left = np.array(part_level_left)
# subpart_level_left = subpart_level_left[ind_max]
# subpart_intensity_left = np.array(part_intensity_left)
# subpart_intensity_left = subpart_intensity_left[ind_max]
# features_left = np.vstack((subpart_intensity_left, subpart_level_left))
#
# # Get particle position and level for right image
# apr_it = apr_right.iterator()
# part_position_right = []
# part_level_right = []
# part_intensity_right = []
# for level in range(apr_it.level_min(), apr_it.level_max()+1):
#     for z in range(apr_it.z_num(level)):
#         for x in range(apr_it.x_num(level)):
#             for idx in range(apr_it.begin(level, z, x), apr_it.end()):
#                 y = apr_it.y(idx)
#                 part_position_right.append([z, y, x])
#                 part_level_right.append(level)
#                 part_intensity_right.append(parts_right[idx])
#
# # Find the Nth particles with highest brightness
# N = 1000
# part_intensity_right = np.array(part_intensity_right)
# ind_max = np.argpartition(part_intensity_right, kth=len(part_intensity_right)-N)
# ind_max = ind_max[-N:]
# subpart_position_right = np.array(part_position_right)
# subpart_position_right = subpart_position_right[ind_max, :]
# subpart_level_right = np.array(part_level_right)
# subpart_level_right = subpart_level_right[ind_max]
# subpart_intensity_right = np.array(part_intensity_right)
# subpart_intensity_right = subpart_intensity_right[ind_max]
# features_right = np.vstack((subpart_intensity_right, subpart_level_right))
#
# # Use Flann method to match particles
# (a, b) = match_features_flann(features_left, features_right, subpart_position_left, subpart_position_right, lowe_ratio=1)
#
# # Add random translation to particles on the right image
# # TODO: add random translation + noise and see if it still works