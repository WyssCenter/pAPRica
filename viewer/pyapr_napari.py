"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pyapr
import napari
from napari.layers import Image, Labels


def apr_to_napari_Image(apr: pyapr.APR,
                        parts: (pyapr.ShortParticles, pyapr.FloatParticles),
                        mode: str = 'constant',
                        level_delta: int = 0,
                        **kwargs):
    """
    Construct a napari 'Image' layer from an APR. Pixel values are reconstructed on the fly via the APRSlicer class.

    Parameters
    ----------
    apr : pyapr.APR
        Input APR data structure
    parts : pyapr.FloatParticles or pyapr.ShortParticles
        Input particle intensities
    mode: str
        Interpolation mode to reconstruct pixel values. Supported values are
            constant:   piecewise constant interpolation
            smooth:     smooth interpolation (via level-adaptive separable smoothing). Note: significantly slower than constant.
            level:      interpolate the particle levels to the pixels
        (default: constant)
    level_delta: int
        Sets the resolution of the reconstruction. The size of the image domain is multiplied by a factor of 2**level_delta.
        Thus, a value of 0 corresponds to the original pixel image resolution, -1 halves the resolution and +1 doubles it.
        (default: 0)

    Returns
    -------
    out : napari.layers.Image
        An Image layer of the APR that can be viewed in napari.
    """
    if 'contrast_limits' in kwargs:
        contrast_limits = kwargs.get('contrast_limits')
        del kwargs['contrast_limits']
    else:
        cmin = apr.level_min() if mode == 'level' else parts.min()
        cmax = apr.level_max() if mode == 'level' else parts.max()
        contrast_limits = [cmin, cmax]
    return Image(data=pyapr.data_containers.APRSlicer(apr, parts, mode=mode, level_delta=level_delta),
                 rgb=False, multiscale=False, contrast_limits=contrast_limits, **kwargs)


def apr_to_napari_Labels(apr: pyapr.APR,
                        parts: pyapr.ShortParticles,
                        mode: str = 'constant',
                        level_delta: int = 0,
                        **kwargs):
    """
    Construct a napari 'Layers' layer from an APR. Pixel values are reconstructed on the fly via the APRSlicer class.

    Parameters
    ----------
    apr : pyapr.APR
        Input APR data structure
    parts : pyapr.FloatParticles or pyapr.ShortParticles
        Input particle intensities
    mode: str
        Interpolation mode to reconstruct pixel values. Supported values are
            constant:   piecewise constant interpolation
            smooth:     smooth interpolation (via level-adaptive separable smoothing). Note: significantly slower than constant.
            level:      interpolate the particle levels to the pixels
        (default: constant)
    level_delta: int
        Sets the resolution of the reconstruction. The size of the image domain is multiplied by a factor of 2**level_delta.
        Thus, a value of 0 corresponds to the original pixel image resolution, -1 halves the resolution and +1 doubles it.
        (default: 0)

    Returns
    -------
    out : napari.layers.Image
        A Labels layer of the APR that can be viewed in napari.
    """
    if 'contrast_limits' in kwargs:
        del kwargs['contrast_limits']

    return Labels(data=pyapr.data_containers.APRSlicer(apr, parts, mode=mode, level_delta=level_delta, tree_mode='max'),
                  multiscale=False, **kwargs)


def display_layers(layers):
    with napari.gui_qt():
        viewer = napari.Viewer()
        for layer in layers:
            viewer.add_layer(layer)
    return viewer

def display_segmentation(apr, parts, mask):
    """
    This function displays an image and its associated segmentation map. It uses napari to lazily generate the pixel
    data from APR on the fly.

    Parameters
    ----------
    apr: (APR) apr object
    parts: (ParticleData) particle object representing the image
    mask: (ParticleData) particle object reprenting the segmentation mask/connected component

    Returns
    -------

    """
    image_nap = apr_to_napari_Image(apr, parts, name='APR')
    mask_nap = apr_to_napari_Labels(apr, mask, name='Segmentation')

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_layer(image_nap)
        viewer.add_layer(mask_nap)


if __name__ == '__main__':
    # Path to APR
    fpath_apr = r'/media/sf_shared_folder_virtualbox/PV_interneurons/output.apr'

    # Instantiate APR and particle objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()  # input particles can be float32 or uint16

    # Read from APR file
    pyapr.io.read(fpath_apr, apr, parts)
    # Multi tile display example
    layers = []
    layers.append(apr_to_napari_Image(apr, parts, name='APR', translate=[0, 0, 0]))
    layers.append(apr_to_napari_Image(apr, parts, name='APR', translate=[0, 0, 2048]))
    layers.append(apr_to_napari_Image(apr, parts, name='APR', translate=[0, 2048, 0]))
    layers.append(apr_to_napari_Image(apr, parts, name='APR', translate=[0, 2048, 2048]))
    # Display APR
    viewer = display_layers(layers)


    # Segmentation display exmaple
    mask = parts > 1500
    display_segmentation(apr, parts, mask)
