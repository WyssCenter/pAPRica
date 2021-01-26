import pyapr
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import napari
from scipy.ndimage import fourier_shift


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

# Parameters
path = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/2P_FF0.7_AF3_Int50_Zoom3_Stacks.tif'
overlap = 25
mpl.use('Qt5Agg')
plt.interactive('on')

# Read image and divide it in two with overlap %
image_ini = imread(path)
image_ini = ((image_ini/4)+100).astype('uint16') # The classifier was train with this intensity transform
s_ini = image_ini.shape
# Image left and image right are overlapping region where we are looking for the
# correct registration parameters
image_left = image_ini[:, :, :int(s_ini[2]/2*(1+overlap/100))]

# Apply a random shift for image_right
np.random.seed(0)
random_displacement = np.random.randn(3)*[1, 5, 5]
image_ini_shifted = np.abs(np.fft.ifftn(fourier_shift(np.fft.fftn(image_ini), shift=random_displacement))).astype('uint16')
image_right = image_ini_shifted[:, :, int(s_ini[2]/2*(1-overlap/100)):]

# Visualize both volumes using Napari with the compensated shift
# display_registration(image_left, image_right, contrast_limit=[0, 40000],
#                      translation=[0, 0, int(s_ini[2]/2*(1-overlap/100))])
# display_registration(image_left, image_right, contrast_limit=[0, 40000],
#                      translation=[0, 0, int(s_ini[2]/2*(1-overlap/100))]-random_displacement)

# Convert volumes to APR
par = pyapr.APRParameters()
par.auto_parameters = False # really heuristic and not working
par.sigma_th = 26.0
par.grad_th = 3.0
par.Ip_th = 253.0
par.rel_error = 0.2
par.gradient_smoothing = 2
apr_left, parts_left = pyapr.converter.get_apr(image_left, verbose=True, params=par)
apr_right, parts_right = pyapr.converter.get_apr(image_right, verbose=True, params=par)

# Perform max-proj on each axis
def get_max_proj(apr, parts, plot=False):
    proj = []
    for d in range(3):
        # dim=0: project along Y to produce a ZX plane
        # dim=1: project along X to produce a ZY plane
        # dim=2: project along Z to produce an XY plane
        proj.append(pyapr.numerics.transform.maximum_projection(apr, parts, dim=d))

    if plot:
        fig, ax = plt.subplots(1, 3)
        for i in range(3):
            ax[i].imshow(proj[i], cmap='gray')
    return proj
proj_left = get_max_proj(apr_left, parts_left, plot=True)
proj_right = get_max_proj(apr_right, parts_right, plot=True)

# Cross-correlate each pair to extract displacement
from skimage.registration import phase_cross_correlation
def get_proj_shifts(proj1, proj2):
    dzx = phase_cross_correlation(proj1[0], proj2[0], return_error=False, upsample_factor=100)
    dzy = phase_cross_correlation(proj1[1], proj2[1], return_error=False, upsample_factor=100)
    dxy = phase_cross_correlation(proj1[2], proj2[2], return_error=False, upsample_factor=100)

    dz = (dzx[0] + dzy[0]) / 2
    dy = (dzy[1] + dxy[1]) / 2
    dx = (dzx[1] + dxy[0]) / 2

    return [dz, dy, dx]

d = get_proj_shifts(proj_left, proj_right)