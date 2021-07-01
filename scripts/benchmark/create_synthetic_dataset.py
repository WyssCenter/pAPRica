"""
Script to create synthetic dataset with multiple CR.

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import numpy as np
import napari
from skimage.morphology import ball
import pyapr
import os
from skimage.io import imsave
from pathlib import Path


def get_coordinates(v, dV, h, dH):
    x = int(v * dV)
    y = int(h * dH)
    x_noise = int(max(v * dV + np.random.randn(1) * 5, 0))
    y_noise = int(max(h * dH + np.random.randn(1) * 5, 0))
    return (x, y, x_noise, y_noise)

# Parameters
path = '/home/apr-benchmark/Desktop/data/synthetic_single_tile'
length = 2048
cell_radius = 5
dH = 4
dV = 4
overlap_H = 25
overlap_V = 25
par = pyapr.APRParameters()
par.auto_parameters = True
par.Ip_th = 120
par.rel_error = 0.2
par.gradient_smoothing = 2
multitile = False

# Create dataset
for n_cells in [128, 512, 2048, 8192, 2**16]:
    # Create synthetic dataset
    data = np.ones([length]*3, dtype='uint16')*100
    cell = ball(cell_radius)*500
    cell[cell==0] = 100
    cell_positions = (np.random.rand(n_cells, 3)*(length-cell.shape[0])).astype('uint16')
    for i in range(n_cells):
        data[
        cell_positions[i, 0]: cell_positions[i, 0]+cell.shape[0],
        cell_positions[i, 1]:cell_positions[i, 1]+cell.shape[0],
        cell_positions[i, 2]:cell_positions[i, 2]+cell.shape[0]
        ] =         data[
        cell_positions[i, 0]: cell_positions[i, 0]+cell.shape[0],
        cell_positions[i, 1]:cell_positions[i, 1]+cell.shape[0],
        cell_positions[i, 2]:cell_positions[i, 2]+cell.shape[0]
        ] + cell

    noise = (np.random.randn(*data.shape)*np.sqrt(data)).astype('uint16')
    data += noise

    # Display synthetic dataset
    # napari.view_image(data)

    if multitile:

        output_folder_apr = os.path.join(path, 'ncells_{}'.format(int(n_cells)))
        Path(output_folder_apr).mkdir(parents=True, exist_ok=True)

        dv = int(data.shape[1] * (1 - overlap_V / 100) / dV)
        dh = int(data.shape[2] * (1 - overlap_H / 100) / dH)


        # Save data as separate tiles
        noise_coordinates = np.zeros((dH * dV, 4))
        for v in range(dV):
            for h in range(dH):
                (x, y, x_noise, y_noise) = get_coordinates(v, dv, h, dh)
                noise_coordinates[v * dH + h, :] = [x_noise, y_noise, x_noise + int(data.shape[1] / dV),
                                                    y_noise + int(data.shape[2] / dH)]

                # Convert data to APR
                apr, parts = pyapr.converter.get_apr(
                    image=data[:, x_noise:x_noise + int(data.shape[1] / dV),
                          y_noise:y_noise + int(data.shape[1] / dV)]
                    , params=par, verbose=False)
                pyapr.io.write(os.path.join(output_folder_apr, '{}_{}.apr'.format(v, h)), apr, parts)

        # Save coordinates
        np.savetxt(os.path.join(output_folder_apr, 'real_displacements.csv'), noise_coordinates, fmt='%1.d',
                   delimiter=',')

    else:
        # Convert data to APR
        apr, parts = pyapr.converter.get_apr(image=data, params=par, verbose=False)
        pyapr.io.write(os.path.join(path, 'ncells_{}.apr'.format(int(n_cells))), apr, parts)
