# In this script I perform cell clustering on nissl data from 2P. First the data is converted to APR
# then segmented and clustered.

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


def scatter_features_2D(x, x_names=None, subsample=1000):
    """
    This function plots each features again each others.
    The total number of plots is therefore n_features*(n_features-1)/2 so condider no using this function
    with high number of features:

    n_features - n_plots
        3      -    3
        4      -    6
        5      -    10
        6      -    15

    Parameters
    ----------
    x: (array) features with shape (n_sample, n_features)
    x_names: (list, optional) list containing the name of features for x/y_labels
    subsample: (int) subsampling factor for plotting

    Returns
    -------
    None
    """
    n_features = x.shape[1]

    for i in range(n_features-1):
        for ii in range(i+1, n_features):
            plt.figure()
            plt.scatter(x[::subsample, i], x[::subsample, ii])
            if x_names is not None:
                plt.xlabel(x_names[i])
                plt.ylabel(x_names[ii])


PROPS = {
    'Area': 'area',
    # 'BoundingBox': 'bbox',
    # 'BoundingBoxArea': 'bbox_area',
    # 'CentralMoments': 'moments_central',
    # 'Centroid': 'centroid',
    # 'ConvexArea': 'convex_area',
    # 'ConvexHull',
    # 'ConvexImage': 'convex_image',
    'Coordinates': 'coords',
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
    # 'WeightedCentroid': 'weighted_centroid',
    # 'WeightedHuMoments': 'weighted_moments_hu',
    # 'WeightedLocalCentroid': 'weighted_local_centroid',
    # 'WeightedMoments': 'weighted_moments',
    # 'WeightedNormalizedMoments': 'weighted_moments_normalized'
}

def get_lmap_from_labels(y_pred, cell_coords, output_shape):
    lmap = np.zeros(output_shape)
    for label, c in zip(y_pred, cell_coords):
        for (x, y, z) in c:
            lmap[x, y, z] = label
    return lmap


def bar_plot(ax, data, f_names=None,  colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


# Parameters
from_image = False
folder_path = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1'
apr_path = r'2P_mouse_re0.2.apr'
image_path = r'stack.tif'
clf_path = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/classifiers/random_forest_n100.joblib'
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working
par.sigma_th = 18162.0
par.grad_th = 3045.0
par.Ip_th = 5351.0
par.rel_error = 0.2
par.gradient_smoothing = 0

if from_image:
    # Load image and transform it to APR
    data = imread(os.path.join(folder_path, image_path))
    data = (data/4+100).astype('uint16')
    apr, parts = pyapr.converter.get_apr(data, verbose=True, params=par)
    print('CR: {:.2f}'.format(data.size/apr.total_number_particles()))
else:
    # Load APR data and classifier
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    pyapr.io.read(os.path.join(folder_path, apr_path), apr, parts)

# Load classifier
clf = load(clf_path)

# Perform segmentation
cc = segment_apr(apr, parts, clf)

# Display segmentation
lmap = np.array(pyapr.numerics.reconstruction.recon_pc(apr, cc), copy=False)
data = np.array(pyapr.numerics.reconstruction.recon_smooth(apr, parts), copy=False)
# display_segmentation(data, lmap)

# Extract cells features
f = pd.DataFrame(regionprops_table(label_image=lmap, intensity_image=data, properties=PROPS.values()))
cells_coords = f['coords']
f.drop(columns=['coords'], inplace=True)

# Perform PCA and display features
from sklearn.decomposition import PCA
pca = PCA(n_components=3, whiten=True)
pca.fit(f)
# plt.plot(pca.explained_variance_ratio_)
f_pca = pca.transform(f)
scatter_features_2D(f_pca, x_names=['Component {}'.format(i) for i in range(3)], subsample=1)

# Perform clustering with Kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(f)
y_pred = kmeans.labels_ + 1
lmap_kmeans = get_lmap_from_labels(y_pred, cells_coords, data.shape)

# Display KMeans results
display_segmentation(data, lmap_kmeans)

f_names = list(f.columns)
centers = kmeans.cluster_centers_/kmeans.cluster_centers_.max(axis=0)
to_plot = {}
for i in range(centers.shape[0]):
    to_plot['Kmeans center {}'.format(i+1)] = centers[i, :]
fig, ax = plt.subplots()
bar_plot(ax, to_plot, f_names)
ax.set_xticks(np.arange(len(f_names)))
ax.set_xticklabels(f_names, rotation='vertical')
fig.tight_layout()

