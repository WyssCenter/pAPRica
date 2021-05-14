"""
This script takes 2 overlapping tiles and the segmentation data and output a list of unique cells.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""
import matplotlib.pyplot as plt
import pyapr
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import napari
from pipapr.viewer import apr_to_napari_Image, apr_to_napari_Labels
import cv2 as cv

def display_cell_centers(apr, parts, cc, cells_overlap, position):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_layer(apr_to_napari_Image(apr, parts, translate=position))
        viewer.add_layer(apr_to_napari_Labels(apr, cc, translate=position))
        viewer.add_points(cells_overlap[['z', 'y', 'x']].to_numpy().astype('uint16'), opacity=0.7)


def display_cell_centers_multiple(apr1, parts1, cc1, cells1, position1, apr2, parts2, cc2, cells2, position2, cells_overlap=None):
    if isinstance(cells1, pd.DataFrame):
        cells1 = cells1[['z', 'y', 'x']].to_numpy().astype('uint16')
    if isinstance(cells2, pd.DataFrame):
        cells2 = cells2[['z', 'y', 'x']].to_numpy().astype('uint16')
    if isinstance(cells_overlap, pd.DataFrame):
        cells_overlap = cells_overlap[['z', 'y', 'x']].to_numpy().astype('uint16')

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_layer(apr_to_napari_Image(apr1, parts1, translate=position1, name='APR 1'))
        viewer.add_layer(apr_to_napari_Labels(apr1, cc1, translate=position1, name='CC 1'))
        viewer.add_points(cells1, opacity=0.7, name='Cells 1')
        viewer.add_layer(apr_to_napari_Image(apr2, parts2, translate=position2, name='APR 2'))
        viewer.add_layer(apr_to_napari_Labels(apr2, cc2, translate=position2, name='CC 2'))
        viewer.add_points(cells2, opacity=0.7, name='Cells 2')
        if cells_overlap is not None:
            viewer.add_points(cells_overlap, opacity=0.7, name='Cells filtered')


def display_merged_cells(apr1, parts1, cc1, position1, apr2, parts2, cc2, position2, cells_merged):
    if isinstance(cells_merged, pd.DataFrame):
        cells_merged = cells_merged[['z', 'y', 'x']].to_numpy().astype('uint16')

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_layer(apr_to_napari_Image(apr1, parts1, translate=position1, name='APR 1'))
        viewer.add_layer(apr_to_napari_Labels(apr1, cc1, translate=position1, name='CC 1'))
        viewer.add_layer(apr_to_napari_Image(apr2, parts2, translate=position2, name='APR 2'))
        viewer.add_layer(apr_to_napari_Labels(apr2, cc2, translate=position2, name='CC 2'))
        viewer.add_points(cells_merged, opacity=0.7, name='Cells merged')


def filter_cells_flann(c1, c2, lowe_ratio=0.7, distance_max=5):

    if lowe_ratio < 0 or lowe_ratio > 1:
        raise ValueError('Lowe ratio is {}, expected between 0 and 1.'.format(lowe_ratio))

    # Match cells descriptors by using Flann method
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.float32(c1), np.float32(c2), k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < lowe_ratio*n.distance and m.distance < distance_max:
            good.append(m)

    # Remove cells that are present in both volumes
    ind_c1 = [m.queryIdx for m in good]
    ind_c2 = [m.trainIdx for m in good]

    # For now I just remove thee cells in c but merging strategies can be better
    c2 = np.delete(c2, ind_c2, axis=0)

    # Display info
    print('{:0.2f}% of cells were removed.'.format(len(ind_c2)/(c1.shape[0]+c2.shape[0]-len(ind_c2))*100))

    return np.vstack((c1, c2))


def merge_cells(cells1, cells2):
    
    # Filter cells to keep only those on the overlapping area
    for i in range(3):
        if i == 0:
            ind = np.where(cells1[:, i] < overlap_i[i])[0]
        else:
            ind = np.concatenate((ind, np.where(cells1[:, i] < overlap_i[i])[0]))
        ind = np.concatenate((ind, np.where(cells1[:, i] > overlap_f[i])[0]))
    ind = np.unique(ind)

    cells1_out = cells1[ind, :]
    cells1_overlap = np.delete(cells1, ind, axis=0)
    
    for i in range(3):
        if i == 0:
            ind = np.where(cells2[:, i] < overlap_i[i])[0]
        else:
            ind = np.concatenate((ind, np.where(cells2[:, i] < overlap_i[i])[0]))
        ind = np.concatenate((ind, np.where(cells2[:, i] > overlap_f[i])[0]))
    ind = np.unique(ind)

    cells2_out = cells2[ind, :]
    cells2_overlap = np.delete(cells2, ind, axis=0)

    cells_filtered_overlap = filter_cells_flann(cells1_overlap, cells2_overlap, lowe_ratio=0.7, distance_max=5)

    cells_merged = np.vstack((cells1_out, cells2_out, cells_filtered_overlap))

    return cells_merged


# Parameters
folder = r'/mnt/Data/wholebrain/multitile/c1'
folder_seg = os.path.join(folder, 'segmentation')

# r in (z, y, x) to be the same dims order as Napari
# r is the position of each tile in the CCF
r1 = np.array([-3, 1192, -25])
r2 = np.array([-2, 1225, 1165])

# Load APRs and CCs
apr1 = pyapr.APR()
apr2 = pyapr.APR()
parts1 = pyapr.ShortParticles()
parts2 = pyapr.ShortParticles()
cc1 = pyapr.LongParticles()
cc2 = pyapr.LongParticles()

pyapr.io.read(os.path.join(folder, '1_0.apr'), apr1, parts1)
pyapr.io.read(os.path.join(folder, '1_1.apr'), apr2, parts2)
pyapr.io.read(os.path.join(folder_seg, '1_0_segmentation.apr'), apr1, cc1)
pyapr.io.read(os.path.join(folder_seg, '1_1_segmentation.apr'), apr2, cc2)
v_size = [apr1.org_dims(2), apr1.org_dims(1), apr1.org_dims(0)]

# Define the overlapping area
overlap_i = r2
overlap_f = np.min((r1 + v_size, r2+v_size), axis=0)

# Retrieve cell centers
cells1 = pyapr.numerics.transform.find_label_centers(apr1, cc1, parts1)
cells1 += r1
cells2 = pyapr.numerics.transform.find_label_centers(apr2, cc2, parts2)
cells2 += r2

cells_merged = merge_cells(cells1, cells2)
display_merged_cells(apr1, parts1, cc1, r1, apr2, parts2, cc2, r2, cells_merged)

# cells1 = pd.DataFrame({'x': cells1[:, 2], 'y': cells1[:, 1], 'z': cells1[:, 0]})
# cells2 = pd.DataFrame({'x': cells2[:, 2], 'y': cells2[:, 1], 'z': cells2[:, 0]})
#
# # Filter cells to keep only those on the overlapping area
# cells1_overlap = cells1.copy()
# cells1_overlap.drop(cells1_overlap[cells1_overlap.x < overlap_i[2]].index, inplace=True)
# cells1_overlap.drop(cells1_overlap[cells1_overlap.x > overlap_f[2]].index, inplace=True)
# cells1_overlap.drop(cells1_overlap[cells1_overlap.y < overlap_i[1]].index, inplace=True)
# cells1_overlap.drop(cells1_overlap[cells1_overlap.y > overlap_f[1]].index, inplace=True)
# cells1_overlap.drop(cells1_overlap[cells1_overlap.z < overlap_i[0]].index, inplace=True)
# cells1_overlap.drop(cells1_overlap[cells1_overlap.z > overlap_f[0]].index, inplace=True)
#
# cells2_overlap = cells2.copy()
# cells2_overlap.drop(cells2_overlap[cells2_overlap.x < overlap_i[2]].index, inplace=True)
# cells2_overlap.drop(cells2_overlap[cells2_overlap.x > overlap_f[2]].index, inplace=True)
# cells2_overlap.drop(cells2_overlap[cells2_overlap.y < overlap_i[1]].index, inplace=True)
# cells2_overlap.drop(cells2_overlap[cells2_overlap.y > overlap_f[1]].index, inplace=True)
# cells2_overlap.drop(cells2_overlap[cells2_overlap.z < overlap_i[0]].index, inplace=True)
# cells2_overlap.drop(cells2_overlap[cells2_overlap.z > overlap_f[0]].index, inplace=True)
#
# cells_filtered_overlap = filter_cells_flann(cells1_overlap, cells2_overlap, lowe_ratio=0.7, distance_max=5)
#
# display_cell_centers_multiple(apr1, parts1, cc1, cells1_overlap, r1, apr2, parts2, cc2, cells2_overlap, r2, cells_filtered_overlap)