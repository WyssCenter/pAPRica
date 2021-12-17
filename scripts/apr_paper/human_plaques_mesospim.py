import numpy as np
import pipapr
import napari
import pyapr
from skimage.exposure import match_histograms
from skimage.filters import gaussian
from time import time
from pipapr.loader import tile_from_apr
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
import seaborn as sns


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


def compute_inplane_gradmag(apr, parts, sobel=True):
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
    dx = pyapr.FloatParticles()
    dy = pyapr.FloatParticles()

    pyapr.numerics.gradient(apr, parts, dx, dimension=1, delta=par.dx, sobel=sobel)
    pyapr.numerics.gradient(apr, parts, dy, dimension=0, delta=par.dy, sobel=sobel)

    dx = np.array(dx, copy=False)
    dy = np.array(dy, copy=False)

    return pyapr.FloatParticles(np.sqrt(dx**2 + dy**2))


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


def map_feature(data, hash_idx, features):
    """
    Map feature values to segmented particle data.

    Parameters
    ----------
    data: (ParticleData) connected component particle array
    hash_idx: (array) array containing the number of each connected component in ascendant order
    features: (array) array containing the values to map

    Returns
    -------
    Array of mapped values
    """

    if len(hash_idx) != len(features):
        raise ValueError('Error: hash_idx and features must have the same length.')

    # Create hash dict
    hash_dict = {x: y for x, y in zip(hash_idx, features)}
    # Replace 0 by 0
    hash_dict[0] = 0

    mp = np.arange(0, data.max() + 1)
    mp[list(hash_dict.keys())] = list(hash_dict.values())
    return mp[np.array(data, copy=False)]


# Parameters
path = '/media/hbm/SSD1/for_segmentation/plaques'

# Convert data
# tiles = pipapr.parser.baseParser(path=path, frame_size=2048, ftype='raw')
# converter = pipapr.converter.tileConverter(tiles)
# converter.batch_convert_to_apr(Ip_th=140, gradient_smoothing=2, rel_error=0.2)

# Parse data
apr_signal, parts_signal = pyapr.io.read('/media/hbm/SSD1/for_segmentation/plaques/APR/568congoredplaques.apr')

autofluo488 = np.fromfile('/media/hbm/SSD1/for_segmentation/plaques/488autofluo.raw', dtype='uint16').reshape((-1, 2048, 2048))
parts_autofluo488 = pyapr.ShortParticles()
parts_autofluo488.sample_image(apr_signal, autofluo488)

autofluo647 = np.fromfile('/media/hbm/SSD1/for_segmentation/plaques/647autofluo.raw', dtype='uint16').reshape((-1, 2048, 2048))
parts_autofluo647 = pyapr.ShortParticles()
parts_autofluo647.sample_image(apr_signal, autofluo647)

parts_autofluo = parts_autofluo647 + parts_autofluo488

# Match histograms
parts_signal = np.array(parts_signal, copy=False)
parts_autofluo = np.array(parts_autofluo, copy=False)
parts_autofluo_hm = match_histograms(parts_autofluo, parts_signal)
# parts_signal = pyapr.ShortParticles(parts_signal)
# parts_autofluo_hm = pyapr.ShortParticles(parts_autofluo_hm)
# viewer = napari.Viewer()
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, colormap='gray', contrast_limits=[0, 2000]))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, pyapr.ShortParticles(parts_autofluo), colormap='gray', contrast_limits=[0, 5000]))
# napari.run()

# Filter signal
parts_filtered = parts_signal.astype('int32') - parts_autofluo_hm.astype('int32')
parts_filtered[parts_filtered<0] = 0
parts_filtered = pyapr.ShortParticles(parts_filtered.astype('uint16'))
# viewer = napari.Viewer()
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, colormap='red', contrast_limits=[0, 2000]))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_filtered, colormap='blue', contrast_limits=[0, 2000]))
# napari.run()

# Resample APR grid on filtered signal
data_filtered = pyapr.numerics.reconstruction.reconstruct_constant(apr_signal, parts_filtered)
param = pyapr.APRParameters()
param.rel_error = 0.2
param.Ip_th = 120
param.gradient_smoothing = 2
apr, parts_filtered = pyapr.converter.get_apr(data_filtered, params=param)

del autofluo647, autofluo488, data_filtered

# Segment plaques
tile = tile_from_apr(apr, parts_filtered)
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

# viewer = napari.Viewer()
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr, parts_signal, contrast_limits=[0, 2000], level_delta=0))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Labels(apr, seg_trainer.parts_cc, level_delta=0))
# napari.run()

# Analyse cmap to obtain information on plaques
plaque_volume = pyapr.numerics.transform.find_label_volume(apr, seg_trainer.parts_cc)
hash_idx = np.arange(0, len(plaque_volume))
hash_idx = hash_idx[plaque_volume>0]
hash_idx = hash_idx[1:]
plaque_volume = plaque_volume[plaque_volume>0]
plaque_volume = plaque_volume[1:]
plaque_volume = plaque_volume * 5 * 5.26 * 5.26
pos = pyapr.numerics.transform.find_label_centers(apr, seg_trainer.parts_cc, weights=parts_filtered)
pos[:, 0] = pos[:, 0] * 5
pos[:, 1] = pos[:, 1] * 5.26
pos[:, 2] = pos[:, 2] * 5.26

# Filter plaques close to the cut edges (depth is hard to predict)
ind = np.where(pos[:, 1]>210*8*5.26)
pos_filtered = np.delete(pos, ind[0], axis=0)
plaque_volume_filtered = np.delete(plaque_volume, ind[0], axis=0)

# Plaque diameter distribution
plt.figure()
plt.hist(plaque_volume, bins=100)
plt.yscale('log')
plt.xlabel('Plaque volume [$\mu m³$]')
plt.ylabel('Counts [#]')

plt.figure()
plaque_size = 2*(3/4*plaque_volume/np.pi)**(1/3)
plaque_size_filtered = 2*(3/4*plaque_volume_filtered/np.pi)**(1/3)
plt.hist(plaque_size, bins=100)
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
parts_mask = compute_gradmag(apr_signal, pyapr.ShortParticles(parts_mask))
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
for i in tqdm(range(pos_filtered.shape[0])):
    current_point = pos_filtered[i, :].reshape((1, 3))
    nd = cdist(current_point, surface_coord)
    plaque_depth.append(nd.min())
plaque_depth = np.array(plaque_depth)

plt.figure()
plt.hist(plaque_depth, bins=100)
# plt.yscale('log')
plt.xlabel('Plaque Depth [$\mu m$]')
plt.ylabel('Counts [#]')
# plt.xlim([0, 500])

# viewer = napari.Viewer()
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, colormap='gray', contrast_limits=[0, 2000]))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr, parts_filtered, colormap='gray', contrast_limits=[0, 2000]))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, pyapr.ShortParticles(parts_autofluo), colormap='gray', contrast_limits=[0, 5000]))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr,
#                                                    pyapr.FloatParticles(map_feature(seg_trainer.parts_cc, hash_idx, plaque_depth)),
#                                                    opacity=0.5, colormap='turbo'))
# napari.run()

# Segment vessel
tile_af = tile_from_apr(apr, parts_autofluo488)
vessel_trainer = pipapr.segmenter.tileTrainer(tile_af,
                                           func_to_compute_features=compute_features)
# vessel_trainer.manually_annotate()
# vessel_trainer.save_labels(path='/media/hbm/SSD1/for_segmentation/plaques/vessel.npy')
vessel_trainer.load_labels(path='/media/hbm/SSD1/for_segmentation/plaques/vessel.npy')
vessel_trainer.train_classifier()
# vessel_trainer.load_classifier(path='/media/hbm/SSD1/for_segmentation/plaques/classifier_vessel.joblib')

prob_list = pipapr.segmenter._predict_on_APR_block(vessel_trainer.f, vessel_trainer.clf, output='proba')
parts_vessel = np.array(prob_list[0], copy=True)
parts_vessel[parts_vessel<50000] = 0


viewer = napari.Viewer()
viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr, parts_filtered, contrast_limits=[0, 2000],
                                                   level_delta=-2, opacity=0.5))
viewer.add_layer(pipapr.viewer.apr_to_napari_Labels(apr, seg_trainer.parts_cc, level_delta=-2, opacity=0.5))
viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr, pyapr.ShortParticles(parts_vessel), contrast_limits=[50000, 2**16],
                                                   level_delta=-2, colormap='red', opacity=0.5))
napari.run()


# Get sample border that match vessel artefact
parts_autofluo_hm = pyapr.ShortParticles(parts_autofluo_hm)
parts_border = parts_autofluo_hm>1000
parts_border = compute_inplane_gradmag(apr_signal, pyapr.ShortParticles(parts_mask))>1
pyapr.numerics.transform.dilation(apr_signal, parts_border, True, 8)

parts_vessel_f = np.array(parts_vessel, copy=True).astype('float64') - np.array(parts_border, copy=True).astype('float64')*2**16
parts_vessel_f[parts_vessel_f<0] = 0
parts_vessel_f = pyapr.ShortParticles(parts_vessel_f.astype('uint16'))

# viewer = napari.Viewer()
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, contrast_limits=[0, 2000],
#                                                    level_delta=0, opacity=1, colormap='gray'))
# # viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal,parts_border,
# #                                                    level_delta=0, opacity=0.5, colormap='green'))
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_vessel_f, contrast_limits=[0, 1],
#                                                    level_delta=0, colormap='red', opacity=0.5))
# napari.run()

patch = pyapr.ReconPatch()
patch.level_delta=-3
mask = pyapr.numerics.reconstruction.reconstruct_smooth(apr_signal, parts_vessel_f, patch=patch)
mask[mask<10000] = 0
mask[mask>0] = 1
vessel_coord = np.array(np.where(mask==1)).T
vessel_coord[:, 0] = vessel_coord[:, 0] * 5 * 8
vessel_coord[:, 1] = vessel_coord[:, 1] * 5.26 * 8
vessel_coord[:, 2] = vessel_coord[:, 2] * 5.26 * 8
plaque_dist_vessel = []
for i in tqdm(range(pos.shape[0])):
    current_point = pos[i, :].reshape((1, 3))
    nd = cdist(current_point, vessel_coord)
    plaque_dist_vessel.append(nd.min())
plaque_dist_vessel = np.array(plaque_dist_vessel)

plt.figure()
plt.hist(plaque_dist_vessel, bins=100)
plt.yscale('log')
plt.xlabel('Plaque distance to nearest vessel [$\mu m$]')
plt.ylabel('Counts [#]')
# plt.xlim([0, 1000])

# Compute distribution of vessel distance
# patch = pyapr.ReconPatch()
# patch.level_delta=-2
# tree_parts = pyapr.LongParticles()
# pyapr.numerics.fill_tree_max(apr_signal, seg_trainer.parts_cc>0, tree_parts)
# gm = pyapr.numerics.reconstruction.reconstruct_constant(apr_signal, seg_trainer.parts_cc>0, patch=patch, tree_parts=tree_parts)
# from skimage.filters import gaussian, threshold_local
# gm = gaussian(gm, sigma=10)
# gm = gm/gm.mean()
# for i in tqdm(range(gm.shape[0])):
#     gm[i] = gm[i] > threshold_local(gm[i], block_size=301, method='mean')
#
# viewer = napari.Viewer()
# viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_filtered, contrast_limits=[0, 2000],
#                                                    level_delta=-2, opacity=0.5))
# # viewer.add_layer(pipapr.viewer.apr_to_napari_Labels(apr_signal, seg_trainer.parts_cc>-2, level_delta=0))
# viewer.add_image(gm, colormap='red', opacity=0.5, scale=[1,1,1], contrast_limits=[0, 1])
# napari.run()
#
# sample_coord = np.array(np.where(mask==1)).T
# sample_coord[:, 0] = sample_coord[:, 0] * 5 * 8
# sample_coord[:, 1] = sample_coord[:, 1] * 5.26 * 8
# sample_coord[:, 2] = sample_coord[:, 2] * 5.26 * 8
# sample_dist_vessel = []
# for i in tqdm(range(sample_coord.shape[0])):
#     current_point = sample_coord[i, :].reshape((1, 3))
#     nd = cdist(current_point, vessel_coord)
#     sample_dist_vessel.append(nd.min())
# sample_dist_vessel = np.array(sample_dist_vessel)
#
# plt.figure()
# plt.hist(plaque_dist_vessel, bins=100, label='Plaque distribution', density=True, alpha=0.5)
# plt.hist(sample_dist_vessel, bins=100, label='Random distribution', density=True, alpha=0.5)
# plt.yscale('log')
# plt.xlabel('Plaque distance to nearest vessel [$\mu m$]')
# plt.ylabel('Normalized density')
# plt.legend()
# # plt.xlim([0, 1000])

plt.scatter(plaque_dist_vessel, plaque_size, c=plaque_depth)
plt.xlabel('Plaque distance to nearest vessel [$\mu m$]')
plt.ylabel('Plaque equivalent diameter [$\mu m$]')
plt.colorbar(label='Plaque depth [$\mu m$]')

plt.scatter(plaque_dist_vessel, plaque_depth)


mask = pyapr.numerics.reconstruction.reconstruct_constant(apr_signal, seg_trainer.parts_cc, patch=patch)
vessel = pyapr.numerics.reconstruction.reconstruct_constant(apr_signal, parts_vessel, patch=patch)
viewer = napari.Viewer()
viewer.add_image(gaussian(vessel, 2), opacity=0.5, colormap='red')
viewer.add_image(gaussian(mask, 2), opacity=0.5, colormap='green')
napari.run()

# Display vessel density and plaque density
patch = pyapr.ReconPatch()
patch.level_delta=-2
plaques = pyapr.numerics.reconstruction.reconstruct_constant(apr_signal, seg_trainer.parts_cc>0, patch=patch)
tata = parts_vessel.copy()
toto = parts_vessel_f>0
pyapr.numerics.transform.dilation(apr_signal, toto, True, 3)
tata[toto==0] = 0
vessel = pyapr.numerics.reconstruction.reconstruct_constant(apr_signal, pyapr.ShortParticles(tata), patch=patch)
vessel[vessel<10000]=0

viewer = napari.Viewer()
viewer.add_image(gaussian(vessel, 5), opacity=0.5, colormap='red')
viewer.add_image(gaussian(plaques, 5), opacity=0.5, colormap='green')
napari.run()


# ***********************************************************************************************
import pandas as pd

df = pd.DataFrame.from_dict({'Plaque equivalent diameter [µm]': plaque_size_filtered,
                             # 'plaque volume': plaque_volume,
                             'Plaque depth [µm]': plaque_depth,
                             # 'plaque dist vessel': plaque_dist_vessel,
                             })

# Figure for the bivariate distribution
sns.jointplot(
    data=df, shade=True, x='Plaque equivalent diameter [µm]', y='Plaque depth [µm]',
    kind="kde"
)

# Viewer for the different sample view (extracted with the file->copy to clipboard option)
viewer = napari.Viewer()
viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, parts_signal, colormap='gray', contrast_limits=[0, 2000]))
viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr, parts_filtered, colormap='gray', contrast_limits=[0, 2000]))
viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr_signal, pyapr.ShortParticles(parts_autofluo), colormap='gray', contrast_limits=[0, 5000]))
viewer.add_layer(pipapr.viewer.apr_to_napari_Image(apr,
                                                   pyapr.FloatParticles(map_feature(seg_trainer.parts_cc, hash_idx, plaque_depth)),
                                                   opacity=0.5, colormap='turbo'))
napari.run()

#
# g = sns.PairGrid(df)
# g.map_lower(sns.kdeplot, fill=True, log_scale=[False, False])
# g.map_upper(sns.kdeplot, fill=True, log_scale=[False, False])
# g.map_diag(sns.histplot, kde=True)

