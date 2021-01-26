import pyapr
import numpy as np
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
import napari
from time import time
from skimage.measure import regionprops
import pandas as pd


def are_labels_the_same(local_labels):
    labels = local_labels[local_labels != 0].flatten()
    return ((labels == labels[0]).all(), labels[0])


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


def display_segmentation(u, lmap, vdim=2):
    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=vdim)
        # add the volume
        viewer.add_image(u, name='Intensity image')
        # add labels
        viewer.add_labels(lmap, name='segmentation')


def predict_on_APR(clf, x):
    # Predict on numpy array
    t = time()
    y_pred = clf.predict(x)
    print('Prediction took {} s.\n'.format(time()-t))

    # Transform numpy array to ParticleData
    parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

    return parts_pred


def predict_and_display(apr, parts, clf, x, save_results=None):
    data = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)
    t = time()
    y_pred = clf.predict(x)
    print('Prediction took {} s.\n'.format(time()-t))
    mask = pyapr.ShortParticles(apr.total_number_particles())
    for i, elem in enumerate(y_pred):
        mask[i] = elem
    labels = np.array(pyapr.numerics.reconstruction.recon_pc(apr, mask), copy=False)
    if save_results is not None:
        imsave(save_results, labels.astype('uint8'), check_contrast=False)
    display_segmentation(data, labels, vdim=2)
    return y_pred


def sample_labels_on_APR(labels, apr, parts_train):

    apr_it = apr.iterator()
    org_dims = apr.org_dims()

    for z in range(apr_it.z_num(apr.level_max())):
        for x in range(apr_it.x_num(apr.level_max())):
            for idx in range(apr_it.begin(apr.level_max(), z, x), apr_it.end()):
                parts_train[idx] = labels[z, x, apr_it.y(idx)]

    # loop over levels up to level_max-1 to assign the remaining labels
    for level in range(apr_it.level_min(), apr_it.level_max()):
        step_size = 2 ** (apr.level_max() - level)  # this is the size (in pixels) of the particle cells at level
        for z in range(apr_it.z_num(level)):
            for x in range(apr_it.x_num(level)):
                for idx in range(apr_it.begin(level, z, x), apr_it.end()):
                    y = apr_it.y(idx)

                    y_start = y * step_size
                    x_start = x * step_size
                    z_start = z * step_size

                    y_end = min(y_start + step_size, org_dims[0])
                    x_end = min(x_start + step_size, org_dims[1])
                    z_end = min(z_start + step_size, org_dims[2])

                    local_labels = labels[z_start:z_end, x_start:x_end, y_start:y_end]

                    if (local_labels == 0).all():  # case where there is no label
                        parts_train[idx] = 0
                    elif np.sum(local_labels != 0) == 1:  # case where there is only one label
                        parts_train[idx] = local_labels.max()
                    else:  # case where there are several labels
                        same, l = are_labels_the_same(local_labels)
                        if same:
                            parts_train[idx] = l
                        else:
                            parts_train[idx] = 0
                            print('Ambiguous label detected, set it to 0.')
    return parts_train

def get_patch(apr, parts, minc, maxc, i, display=False):
    image_patch = np.array(pyapr.numerics.reconstruction.recon_patch(apr, parts,
                                                            minc[i, 0], maxc[i, 0],
                                                            minc[i, 1], maxc[i, 1],
                                                            minc[i, 2], maxc[i, 2]))
    if display:
        with napari.gui_qt():
            viewer = napari.Viewer(ndisplay=2)
            # add the volume
            viewer.add_image(image_patch, name='Patch')

    return image_patch

PROPS = {
    'Area': 'area',
    'BoundingBox': 'bbox',
    'BoundingBoxArea': 'bbox_area',
    'CentralMoments': 'moments_central',
    'Centroid': 'centroid',
    'ConvexArea': 'convex_area',
    # 'ConvexHull',
    'ConvexImage': 'convex_image',
    'Coordinates': 'coords',
    # 'Eccentricity': 'eccentricity',
    'EquivDiameter': 'equivalent_diameter',
    'EulerNumber': 'euler_number',
    'Extent': 'extent',
    # 'Extrema',
    # 'FeretDiameterMax': 'feret_diameter_max',
    'FilledArea': 'filled_area',
    'FilledImage': 'filled_image',
    # 'HuMoments': 'moments_hu',
    'Image': 'image',
    'InertiaTensor': 'inertia_tensor',
    'InertiaTensorEigvals': 'inertia_tensor_eigvals',
    'IntensityImage': 'intensity_image',
    'Label': 'label',
    'LocalCentroid': 'local_centroid',
    'MajorAxisLength': 'major_axis_length',
    'MaxIntensity': 'max_intensity',
    'MeanIntensity': 'mean_intensity',
    'MinIntensity': 'min_intensity',
    'MinorAxisLength': 'minor_axis_length',
    'Moments': 'moments',
    'NormalizedMoments': 'moments_normalized',
    # 'Orientation': 'orientation',
    # 'Perimeter': 'perimeter',
    # 'CroftonPerimeter': 'perimeter_crofton',
    # 'PixelIdxList',
    # 'PixelList',
    # 'Slice': 'slice',
    # 'Solidity': 'solidity',
    # 'SubarrayIdx'
    'WeightedCentralMoments': 'weighted_moments_central',
    'WeightedCentroid': 'weighted_centroid',
    # 'WeightedHuMoments': 'weighted_moments_hu',
    'WeightedLocalCentroid': 'weighted_local_centroid',
    'WeightedMoments': 'weighted_moments',
    'WeightedNormalizedMoments': 'weighted_moments_normalized'
}

OBJECT_COLUMNS = {
    'image', 'coords', 'convex_image', 'slice',
    'filled_image', 'intensity_image'
}

COL_DTYPES = {
    'area': int,
    'bbox': int,
    'bbox_area': int,
    'moments_central': float,
    'centroid': float,
    'convex_area': int,
    'convex_image': object,
    'coords': object,
    'eccentricity': float,
    'equivalent_diameter': float,
    'euler_number': int,
    'extent': float,
    'feret_diameter_max': float,
    'filled_area': int,
    'filled_image': object,
    'moments_hu': float,
    'image': object,
    'inertia_tensor': float,
    'inertia_tensor_eigvals': float,
    'intensity_image': object,
    'label': int,
    'local_centroid': float,
    'major_axis_length': float,
    'max_intensity': int,
    'mean_intensity': float,
    'min_intensity': int,
    'minor_axis_length': float,
    'moments': float,
    'moments_normalized': float,
    'orientation': float,
    'perimeter': float,
    'perimeter_crofton': float,
    'slice': object,
    'solidity': float,
    'weighted_moments_central': float,
    'weighted_centroid': float,
    'weighted_moments_hu': float,
    'weighted_local_centroid': float,
    'weighted_moments': float,
    'weighted_moments_normalized': float
}

PROP_VALS = set(PROPS.values())


def _props_to_dict(regions, properties=PROPS.values(), separator='-'):

    out = {}
    n = len(regions)
    for prop in properties:
        r = regions[0]
        rp = getattr(r, prop)
        dtype = COL_DTYPES[prop]
        column_buffer = np.zeros(n, dtype=dtype)

        # scalars and objects are dedicated one column per prop
        # array properties are raveled into multiple columns
        # for more info, refer to notes 1
        if np.isscalar(rp) or prop in OBJECT_COLUMNS or dtype is np.object_:
            for i in range(n):
                column_buffer[i] = regions[i][prop]
            out[prop] = np.copy(column_buffer)
        else:
            if isinstance(rp, np.ndarray):
                shape = rp.shape
            else:
                shape = (len(rp),)

            for ind in np.ndindex(shape):
                for k in range(n):
                    loc = ind if len(ind) > 1 else ind[0]
                    column_buffer[k] = regions[k][prop][loc]
                modified_prop = separator.join(map(str, (prop,) + ind))
                out[modified_prop] = np.copy(column_buffer)
    return out


# APR file to segment
fpath_apr = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/2P_mouse_re0.2.apr'
fpath_labels = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/manual_sparse_labels_membrane.npy'

# Instantiate APR and particle objects
apr = pyapr.APR()
parts = pyapr.ShortParticles()  # input particles can be float32 or uint16
labels = np.load(fpath_labels).squeeze().astype('uint16') # 0: not labeled - 1: cells - 2: background - 3: membrane

# Read from APR file
pyapr.io.read(fpath_apr, apr, parts)

# Sample labels on APR grid
parts_train_C = pyapr.ShortParticles(apr.total_number_particles())
parts_train_C = sample_labels_on_APR(labels, apr, parts_train_C)
parts_train = np.array(parts_train_C, copy=False)

# Display sampled particles
# pyapr.viewer.parts_viewer(apr, parts_train_C)

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
print('Features computation took {} s.'.format(time()-t))

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
plt.imshow(np.corrcoef(f.T), cmap='jet')
plt.colorbar()
plt.xticks(np.arange(len(f_names)), f_names, rotation=45)
plt.yticks(np.arange(len(f_names)), f_names)
plt.tight_layout()

# Fetch data that was manually labelled
ind_manual_label = (parts_train != 0)
x = f[ind_manual_label, :]
y = parts_train[ind_manual_label]-1

# Train SVM classification
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

clf = make_pipeline(preprocessing.StandardScaler(with_mean=True, with_std=True),
                    RandomForestClassifier(n_estimators=10, class_weight='balanced'))
t = time()
clf.fit(x, y)
print('Training took {} s.\n'.format(time()-t))

# Display tree
# from sklearn import tree
# tree.plot_tree(clf[1].estimators_[0],
#                feature_names=f_names,
#                class_names=['Cells', 'Membranes', 'Background'],
#                filled=True)

# Apply on whole dataset and display results
# save_results = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/segmentation_results/labels.tif'
# Predict particle type (cell, membrane or brackground) for each cell with the trained model
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

# Display results with napari
lmap = np.array(pyapr.numerics.reconstruction.recon_pc(apr, cc), copy=False)
data = np.array(pyapr.numerics.reconstruction.recon_smooth(apr, parts), copy=False)
display_segmentation(data, lmap)

# Save classifier
# Note: this method is not perfect and it might break if scikit-learn version is not the same between the
# dump and the loading. Use with caution (see https://scikit-learn.org/stable/modules/model_persistence.html)
from joblib import dump
dump(clf, '/media/sf_shared_folder_virtualbox/mouse_2P/data1/classifiers/random_forest_n100.joblib')
# To load back use:
# from joblib import load
# toto = load('/media/sf_shared_folder_virtualbox/mouse_2P/data1/classifiers/random_forest_n100.joblib')

# Get labeled objects patch coordinates (some id do not appear in cc)
# minc, maxc = pyapr.numerics.transform.find_objects(apr, cc)
# valid_labels = [(minc[i, :] < maxc[i, :]).all() for i in range(minc.shape[0])]
# minc = minc[valid_labels, :]
# maxc = maxc[valid_labels, :]
#
# # Compute the number of detected cells
# print('Total number of detected cells: {}\n'.format(len(valid_labels)))
#
# # Loop to reconstruct object and compute properties on them
# mask = cc>0
# n_object = minc.shape[0]
# cells = []
# for i in range(n_object):
#     image_patch = get_patch(apr, parts, minc, maxc, i)
#     label_patch = get_patch(apr, mask, minc, maxc, i)
#     l = regionprops(label_image=label_patch, intensity_image=image_patch)
#     if len(l)>1:
#         raise ValueError('Several cells detected in the patch.')
#     cells.append(l[0])
#
# cells_dataframe = pd.DataFrame(cells)

