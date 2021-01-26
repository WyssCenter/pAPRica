# This approach does not work because the feature space is too small and flann method have a hard time
# finding good matches because the second neighbor is always close (with a higher
# number of feature this should not happen)

import pyapr
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import os
import napari
import cv2 as cv

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



# Parameters
path = r'/home/jules/Desktop/data/cells_3D_raw.tif'
overlap = 25

# Read image and divide it in two with overlap %
image_ini = imread(path)
s_ini = image_ini.shape
# Image left and image right are overlapping region where we are looking for the
# correct registration parameters
image_left = image_ini[:, :, :int(s_ini[2]/2*(1+overlap/100))]
image_right = image_ini[:, :, int(s_ini[2]/2*(1-overlap/100)):]

# Visualize both volumes using Napari
# display_registration(image_left, image_right, translation=[0, 0, int(s_ini[2]/2*(1-overlap/100))])

# Convert volumes to APR
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working
par.sigma_th = 26.0
par.grad_th = 3.0
par.Ip_th = 253.0
par.rel_error = 0.4
par.gradient_smoothing = 2
apr_left, parts_left = pyapr.converter.get_apr(image_left, verbose=True, params=par)
apr_right, parts_right = pyapr.converter.get_apr(image_right, verbose=True, params=par)

# Create sub-images on the overlapping area
parts_left = np.array(parts_left)
parts_right = np.array(parts_right)

# Get particle position and level for left image
apr_it = apr_left.iterator()
part_position_left = []
part_level_left = []
part_intensity_left = []
for level in range(apr_it.level_min(), apr_it.level_max()+1):
    for z in range(apr_it.z_num(level)):
        for x in range(apr_it.x_num(level)):
            for idx in range(apr_it.begin(level, z, x), apr_it.end()):
                y = apr_it.y(idx)
                part_position_left.append([z, y, x])
                part_level_left.append(level)
                part_intensity_left.append(parts_left[idx])

# Find the Nth particles with highest brightness
N = 1000
part_intensity_left = np.array(part_intensity_left)
ind_max = np.argpartition(part_intensity_left, kth=len(part_intensity_left)-N)
ind_max = ind_max[-N:]
subpart_position_left = np.array(part_position_left)
subpart_position_left = subpart_position_left[ind_max, :]
subpart_level_left = np.array(part_level_left)
subpart_level_left = subpart_level_left[ind_max]
subpart_intensity_left = np.array(part_intensity_left)
subpart_intensity_left = subpart_intensity_left[ind_max]
features_left = np.vstack((subpart_intensity_left, subpart_level_left))

# Get particle position and level for right image
apr_it = apr_right.iterator()
part_position_right = []
part_level_right = []
part_intensity_right = []
for level in range(apr_it.level_min(), apr_it.level_max()+1):
    for z in range(apr_it.z_num(level)):
        for x in range(apr_it.x_num(level)):
            for idx in range(apr_it.begin(level, z, x), apr_it.end()):
                y = apr_it.y(idx)
                part_position_right.append([z, y, x])
                part_level_right.append(level)
                part_intensity_right.append(parts_right[idx])

# Find the Nth particles with highest brightness
N = 1000
part_intensity_right = np.array(part_intensity_right)
ind_max = np.argpartition(part_intensity_right, kth=len(part_intensity_right)-N)
ind_max = ind_max[-N:]
subpart_position_right = np.array(part_position_right)
subpart_position_right = subpart_position_right[ind_max, :]
subpart_level_right = np.array(part_level_right)
subpart_level_right = subpart_level_right[ind_max]
subpart_intensity_right = np.array(part_intensity_right)
subpart_intensity_right = subpart_intensity_right[ind_max]
features_right = np.vstack((subpart_intensity_right, subpart_level_right))

# Use Flann method to match particles
(a, b) = match_features_flann(features_left, features_right, subpart_position_left, subpart_position_right, lowe_ratio=1)

# Add random translation to particles on the right image
# TODO: add random translation + noise and see if it still works


