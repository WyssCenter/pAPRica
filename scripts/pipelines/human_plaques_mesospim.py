import numpy as np
import pipapr
import napari
import pyapr
from skimage.exposure import match_histograms
from time import time
from pipapr.loader import tile_from_apr
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist


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
    t = time()
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


def get_cc_from_features(apr, parts_pred):

    # Create a mask from particle classified as cells (cell=1, background=2, membrane=3)
    parts_cells = (parts_pred == 1)

    # Use opening to separate touching cells
    pyapr.numerics.transform.opening(apr, parts_cells, binary=True, inplace=True)

    # Apply connected component
    cc = pyapr.LongParticles()
    pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)

    # # Remove small and large objects
    pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=5*5*5)
    # pyapr.numerics.transform.remove_large_objects(apr, cc, max_volume=128)

    return cc

# Parameters
path = '/media/hbm/SSD1/for_segmentation/plaques'

# Convert data
# tiles = pipapr.parser.randomParser(path=path, frame_size=2048, ftype='raw')
# converter = pipapr.converter.tileConverter(tiles)
# converter.batch_convert_to_apr(Ip_th=140, gradient_smoothing=2, rel_error=0.2)

# Parse data
apr_signal, parts_signal = pyapr.io.read('/media/hbm/SSD1/for_segmentation/plaques/APR/568congoredplaques.apr')

autofluo488 = np.fromfile('/media/hbm/SSD1/for_segmentation/plaques/488autofluo.raw', dtype='uint16').reshape((-1, 2048, 2048))
parts_autofluo488 = pyapr.ShortParticles()
parts_autofluo488.sample_image(apr_signal, autofluo488)

autofluo488 = np.fromfile('/media/hbm/SSD1/for_segmentation/plaques/647autofluo.raw', dtype='uint16').reshape((-1, 2048, 2048))
parts_autofluo647 = pyapr.ShortParticles()
parts_autofluo647.sample_image(apr_signal, autofluo488)

parts_autofluo = parts_autofluo647 + parts_autofluo488

# Match histograms
parts_signal = np.array(parts_signal, copy=False)
parts_autofluo = np.array(parts_autofluo, copy=False)
parts_autofluo_hm = match_histograms(parts_autofluo, parts_signal)
# parts_signal = pyapr.ShortParticles(parts_signal)
# parts_autofluo_hm = pyapr.ShortParticles(parts_autofluo_hm)
# viewer = napari.Viewer()
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, colormap='red', contrast_limits=[0, 1000]))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_autofluo_hm, colormap='green', contrast_limits=[0, 1000]))
# napari.run()

# Filter signal
parts_filtered = parts_signal.astype('int32') - parts_autofluo_hm.astype('int32')
parts_filtered[parts_filtered<0] = 0
parts_filtered = pyapr.ShortParticles(parts_filtered.astype('uint16'))
# viewer = napari.Viewer()
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, colormap='red', contrast_limits=[0, 2000]))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_filtered, colormap='blue', contrast_limits=[0, 2000]))
# napari.run()

# Segment plaques
tile = tile_from_apr(apr_signal, parts_filtered)
seg_trainer = pipapr.segmenter.tileTrainer(tile,
                                           func_to_compute_features=compute_features,
                                           func_to_get_cc=get_cc_from_features)
# seg_trainer.manually_annotate()
# seg_trainer.save_labels(path='/media/hbm/SSD1/for_segmentation/plaques/APR')
seg_trainer.load_labels(path='/media/hbm/SSD1/for_segmentation/plaques/plaques.npy')
seg_trainer.train_classifier()
# seg_trainer.save_classifier(path='/media/hbm/SSD1/for_segmentation/plaques/classifier.joblib')

seg_trainer.segment_training_tile(bg_label=1)
parts_signal = pyapr.ShortParticles(parts_signal)

viewer = napari.Viewer()
viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, contrast_limits=[0, 2000], level_delta=0))
viewer.add_layer(pipapr.viewer.apr_to_napari_Labels(apr_signal, seg_trainer.parts_cc, level_delta=0))
napari.run()

# Analyse cmap to obtain information on plaques
vol = pyapr.numerics.transform.find_label_volume(apr_signal, seg_trainer.parts_cc)
vol = vol[vol>0]
vol = vol[1:]
vol = vol * 5 * 5.26 * 5.26
pos = pyapr.numerics.transform.find_label_centers(apr_signal, seg_trainer.parts_cc, weights=parts_filtered)
pos[:, 0] = pos[:, 0] * 5
pos[:, 1] = pos[:, 1] * 5.26
pos[:, 2] = pos[:, 2] * 5.26

# Plaque diameter distribution
plt.figure()
plt.hist(vol, bins=100)
plt.yscale('log')
plt.xlabel('Plaque volume [$\mu m³$]')
plt.ylabel('Counts [#]')

plt.figure()
diam = 2*(3/4*vol/np.pi)**(1/3)
plt.hist(diam, bins=100)
plt.xlabel('Plaque equivalent diameter [$\mu m$]')
plt.ylabel('Counts [#]')

# Plaque nearest neighbors analysis
nn_distance = []
for i in tqdm(range(pos.shape[0])):
    current_point = pos[i, :].reshape((1, 3))
    nd = cdist(current_point, pos)
    nd = nd[nd>0]
    nn_distance.append(nd.min())
nn_distance = np.array(nn_distance)

plt.figure()
plt.hist(nn_distance, bins=100)
plt.yscale('log')
plt.xlabel('Nearest plaque [$\mu m$]')
plt.ylabel('Counts [#]')
plt.xlim([0, 500])

# Segment brain complete volume
parts_mask = parts_autofluo_hm>250
parts_mask = compute_gradmag(apr_signal, parts_mask)
parts_mask = parts_mask*1000
parts_mask = np.array(parts_mask, copy=False)
parts_mask = parts_mask.astype('uint16')
parts_mask = pyapr.ShortParticles(parts_mask)
patch = pyapr.ReconPatch()
patch.level_delta=-3
mask = pyapr.numerics.reconstruction.reconstruct_constant(apr_signal, parts_mask, patch=patch)
mask[mask<120] = 0
mask[mask>0] = 1
mask = mask.astype('bool')
mask[:, :, 0:119] = 0

# Compute plaque depth
surface_coord = np.array(np.where(mask==1)).T
surface_coord[:, 0] = surface_coord[:, 0] * 5 * 8
surface_coord[:, 1] = surface_coord[:, 1] * 5.26 * 8
surface_coord[:, 2] = surface_coord[:, 2] * 5.26 * 8
plaque_depth = []
for i in tqdm(range(pos.shape[0])):
    current_point = pos[i, :].reshape((1, 3))
    nd = cdist(current_point, surface_coord)
    plaque_depth.append(nd.min())
plaque_depth = np.array(plaque_depth)

plt.figure()
plt.hist(plaque_depth, bins=100)
# plt.yscale('log')
plt.xlabel('Plaque Depth [$\mu m$]')
plt.ylabel('Counts [#]')
# plt.xlim([0, 500])

pld = np.zeros_like(mask, dtype='uint16')
for i in tqdm(range(pos.shape[0])):
    current_point = pos[i, :]
    z = int(current_point[0]/(5*8))
    y = int(current_point[1]/(5.26*8))
    x = int(current_point[2]/(5.26*8))
    pld[z, y, x] = int(plaque_depth[i])

viewer = napari.Viewer()
viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, contrast_limits=[0, 2000], level_delta=-3))
viewer.add_image(pld)
napari.run()

# Correlation analysis
f = np.vstack((plaque_depth, diam, nn_distance))
ind = nn_distance<400
ind = np.tile(ind, (3,1))
f = f[ind].reshape((3, -1))