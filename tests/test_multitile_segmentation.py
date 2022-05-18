import numpy as np
from skimage.morphology import ball
import pyapr
import os
from skimage.io import imsave
from pathlib import Path
import pipapr


def get_coordinates(v, dV, h, dH):
    x = int(v * dV)
    y = int(h * dH)
    x_noise = int(max(v * dV + np.random.randn(1) * 5, 0))
    y_noise = int(max(h * dH + np.random.randn(1) * 5, 0))
    return (x, y, x_noise, y_noise)

# Parameters
path = './data/synthetic'
length = 2048
cell_radius = 5
dH = 4
dV = 4
overlap_H = 25
overlap_V = 25
n_cells = 2048

Path(os.path.join(path, 'tif')).mkdir(parents=True, exist_ok=True)

# Create synthetic dataset
data = np.ones([length]*3, dtype='uint16')*100
cell = ball(cell_radius)*500
cell[cell==0] = 100
cell_positions = (np.random.rand(n_cells, 3)*(length-cell.shape[0])).astype('uint16')
label = np.zeros_like(data)
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

    label[
    cell_positions[i, 0]: cell_positions[i, 0] + cell.shape[0],
    cell_positions[i, 1]:cell_positions[i, 1] + cell.shape[0],
    cell_positions[i, 2]:cell_positions[i, 2] + cell.shape[0]
    ] = 1


noise = (np.random.randn(*data.shape)*np.sqrt(data)).astype('uint16')
data += noise

dv = int(data.shape[1] * (1 - overlap_V / 100) / dV)
dh = int(data.shape[2] * (1 - overlap_H / 100) / dH)


# Save data as separate tiles
noise_coordinates = np.zeros((dH * dV, 4))
for v in range(dV):
    for h in range(dH):
        (x, y, x_noise, y_noise) = get_coordinates(v, dv, h, dh)
        noise_coordinates[v * dH + h, :] = [x_noise, y_noise, x_noise + int(data.shape[1] / dV),
                                            y_noise + int(data.shape[2] / dH)]

        # Save data
        imsave(os.path.join(path, '{}_{}.tif'.format(v, h)),
               data[:, x_noise:x_noise + int(data.shape[1] / dV), y_noise:y_noise + int(data.shape[1] / dV)],
               check_contrast=False)

        # Get labels for first tile
        if (v == 0) and (h == 0):
            label = label[:, x_noise:x_noise + int(data.shape[1] / dV), y_noise:y_noise + int(data.shape[1] / dV)]

# Save coordinates
np.savetxt(os.path.join(path, 'real_displacements.csv'), noise_coordinates, fmt='%1.d',
           delimiter=',')

# Parse synthetic data
tiles = pipapr.tileParser(os.path.join(path, 'tif'))
tiles.check_files_integrity()

# Convert them to APR
converter = pipapr.converter.tileConverter(tiles)
converter.batch_convert_to_apr(Ip_th=100, rel_error=0.4, path=os.path.join(path, 'APR'))

# Parse synthetic data
tiles_apr = pipapr.tileParser(os.path.join(path, 'APR'))
tiles_apr.compute_average_CR()
tiles_apr.check_files_integrity()

# Simulate training of segmentation
tile = tiles_apr[0]
tile.load_tile()
parts_label = tile.apr.sample_image(label)
trainer = pipapr.tileTrainer(tile)