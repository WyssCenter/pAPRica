import numpy as np
from skimage.morphology import ball
import pyapr
import os
from skimage.io import imsave
from pathlib import Path
import pipapr
import sparse


def get_coordinates(v, dV, h, dH):
    x = int(v * dV)
    y = int(h * dH)
    x_noise = int(max(v * dV + np.random.randn(1) * 5, 0))
    y_noise = int(max(h * dH + np.random.randn(1) * 5, 0))
    return (x, y, x_noise, y_noise)


def compute_gradients(apr, parts):
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

    dz = pyapr.filter.gradient(apr, parts, dim=2, delta=par.dz)
    dx = pyapr.filter.gradient(apr, parts, dim=1, delta=par.dx)
    dy = pyapr.filter.gradient(apr, parts, dim=0, delta=par.dy)
    return dz, dx, dy


def compute_laplacian(apr, parts, grad=None):
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
        dz, dx, dy = compute_gradients(apr, parts)
    else:
        dz, dx, dy = grad
    dz2 = pyapr.filter.gradient(apr, dz, dim=2, delta=par.dz)
    dx2 = pyapr.filter.gradient(apr, dx, dim=1, delta=par.dx)
    dy2 = pyapr.filter.gradient(apr, dy, dim=0, delta=par.dy)
    return dz2 + dx2 + dy2


def compute_gradmag(apr, parts):
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
    gradmag = pyapr.filter.gradient_magnitude(apr, parts, deltas=(par.dz, par.dx, par.dy))
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

    stencil = pyapr.filter.get_gaussian_stencil(size, sigma, ndims=3, normalize=True)
    return pyapr.filter.convolve(apr, parts, stencil)


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
    # Compute lvl for each particle
    lvl = particle_levels(apr)
    print('Particle level computed.')
    # Compute difference of Gaussian
    dog = gaussian_blur(apr, parts, sigma=3, size=22) - gauss
    print('DOG computed.')
    lapl_of_gaussian = compute_laplacian(apr, gauss)
    print('Laplacian of Gaussian computed.')

    # Aggregate filters in a feature array
    f = np.vstack((np.array(parts, copy=True),
                   lvl,
                   gauss,
                   grad,
                   lapl_of_gaussian,
                   dog
                   )).T

    return f


def get_cc_from_features(apr, parts_pred):

    # Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
    parts_cells = (parts_pred == 2)

    # Use opening to separate touching cells
    pyapr.morphology.opening(apr, parts_cells, radius=1, binary=True, inplace=True)

    # Apply connected component
    cc = pyapr.LongParticles()
    pyapr.measure.connected_component(apr, parts_cells, cc)

    # Remove small objects
    # cc = pyapr.numerics.transform.remove_small_objects(apr, cc, 128)

    return cc


# Parameters
path = './data/synthetic'
length = 1024
cell_radius = 5
dH = 4
overlap_H = 25
overlap_V = 25
n_cells = 2048

dV = dH
Path(os.path.join(path, 'tif')).mkdir(parents=True, exist_ok=True)

# Create synthetic dataset
data = np.ones([length]*3, dtype='uint16')*100
cell = ball(cell_radius)*500
cell[cell==0] = 100
cell_positions = (np.random.rand(n_cells, 3)*(length-cell.shape[0])).astype('uint16')
label = np.ones_like(data, dtype='uint16')
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
    ] = 2


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
        imsave(os.path.join(path, 'tif', '{}_{}.tif'.format(v, h)),
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

# Convert them to APR
converter = pipapr.converter.tileConverter(tiles)
converter.batch_convert_to_apr(Ip_th=100, rel_error=0.4, path=os.path.join(path, 'APR'))

# Parse synthetic data
tiles_apr = pipapr.tileParser(os.path.join(path, 'APR'), frame_size=int(length/dH))

# Stitch data-set
stitcher = pipapr.tileStitcher(tiles_apr, overlap_h=overlap_H, overlap_v=overlap_V)
stitcher.compute_registration()

# Simulate training of segmentation
tile = tiles_apr[0]
tile.load_tile()
trainer = pipapr.tileTrainer(tile, func_to_get_cc=get_cc_from_features, func_to_compute_features=compute_features)
trainer.labels_manual = sparse.COO.from_numpy(label[:100, :100, :100])
trainer.pixel_list = trainer.labels_manual.coords.T
trainer.labels = trainer.labels_manual.data
trainer.train_classifier(n_estimators=100)

# Segment tiles
segmenter = pipapr.multitileSegmenter.from_trainer(tiles_apr, database=stitcher.database, trainer=trainer)
segmenter.compute_multitile_segmentation(save_cc=False)