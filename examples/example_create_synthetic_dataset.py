"""
Script to create a synthetic multi-tile dataset.


By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import numpy as np
from skimage.morphology import ball
from skimage.io import imsave
import pyapr
import os
from pathlib import Path


def get_coordinates(v, dV, h, dH):
    x = int(v * dV)
    y = int(h * dH)
    x_noise = int(max(v * dV + np.random.randn(1) * 5, 0))
    y_noise = int(max(h * dH + np.random.randn(1) * 5, 0))
    return x, y, x_noise, y_noise


# Parameters
output_path = 'path_to_folder'  # Folder where the data will be saved
type = 'apr'                    # 'apr' or 'tiff'
length = 2048                   # Size of the generated data, it will be of shape (length, length, length) so
                                # increase with caution
n_cells = 2048                  # Number of object in the dataset
cell_radius = 5                 # Size of the objects
dH = 4                          # Number of tiles horizontally
dV = 4                          # Number of tiles vertically
overlap_H = 25                  # Horizontal overlap
overlap_V = 25                  # Vertical overlap

# Parameters you should not modify
par = pyapr.APRParameters()
par.auto_parameters = True
par.Ip_th = 120
par.rel_error = 0.2
par.gradient_smoothing = 2

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

output_folder_apr = os.path.join(output_path, 'ncells_{}'.format(int(n_cells)))
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

        if type == 'apr':
            # Convert data to APR
            apr, parts = pyapr.converter.get_apr(image=data[:, x_noise:x_noise + int(data.shape[1] / dV),
                                                       y_noise:y_noise + int(data.shape[1] / dV)],
                                                 params=par,
                                                 verbose=False)
            pyapr.io.write(os.path.join(output_folder_apr, '{}_{}.apr'.format(v, h)), apr, parts)
        elif type == 'tiff':
            imsave(os.path.join(output_folder_apr, '{}_{}.apr'.format(v, h)),
                   data[:, x_noise:x_noise + int(data.shape[1] / dV),
                   y_noise:y_noise + int(data.shape[1] / dV)])
        else:
            raise TypeError('Error: type should be either \'apr\' or \'tiff\'.')

# Save coordinates
np.savetxt(os.path.join(output_folder_apr, 'real_displacements.csv'), noise_coordinates, fmt='%1.d',
           delimiter=',')