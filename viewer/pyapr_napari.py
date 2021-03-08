import pyapr
import numpy as np
import napari
from napari.layers import Image, Labels
from numbers import Integral
from time import time

class APRArray:
    def __init__(self, apr, parts, type='constant'):
        self.apr = apr
        self.parts = parts
        self.dims = apr.org_dims()
        self.ndim = 3
        self.shape = [self.dims[2], self.dims[1], self.dims[0]]
        self.dtype = np.float32 if isinstance(parts, pyapr.FloatParticles) else np.uint16
        self.t_last_call = []
        self.t_getitem = []

        if type == 'constant':
            self.recon = pyapr.numerics.reconstruction.recon_patch
        elif type == 'smooth':
            self.recon = pyapr.numerics.reconstruction.recon_patch_smooth
        elif type == 'level':
            raise NotImplementedError()  # this can be implemented
        else:
            raise ValueError('APRArray type must be \'constant\' or \'smooth\'')

    def _getarray(self, *args):
        drange = [-1, -1, -1, -1, -1, -1]  # [z_min, z_max, y_min, y_max, x_min, x_max]
        for i in range(len(args)):
            if args[i] is not None:
                drange[i] = args[i]
        return np.array(pyapr.numerics.reconstruction.recon_patch(self.apr, self.parts, *drange)).squeeze()

    def __getitem__(self, item):
        if isinstance(item, Integral):
            return self._getarray(item, item+1)
        elif isinstance(item, slice):
            return self._getarray(item.start, item.stop)
        elif isinstance(item, (tuple, list)):
            if len(item) > 3:
                raise ValueError('nope')
            drange = [-1, -1, -1, -1, -1, -1]
            for i in range(len(item)):
                if isinstance(item[i], Integral):
                    drange[2*i] = item[i]
                    drange[2*i+1] = item[i]+1
                elif isinstance(item[i], slice):
                    drange[2*i] = item[i].start
                    drange[2*i+1] = item[i].stop
                else:
                    raise ValueError('got item of type {}'.format(type(item[i])))
            return self._getarray(*drange)
        else:
            raise ValueError('got item of type {}'.format(type(item)))


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
    aprarr = APRArray(apr, parts, type='constant')
    image_nap = Image(data=aprarr, rgb=False, multiscale=False, name='APR')
    maskarr = APRArray(apr, mask, type='constant')
    mask_nap = Labels(data=maskarr, multiscale=False, name='Segmentation')

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
    aprarr = APRArray(apr, parts, type='constant')

    # Multi tile display example
    layers = []
    layers.append(Image(data=aprarr, rgb=False, multiscale=False, name='APR', translate=[0, 0, 0]))
    layers.append(Image(data=aprarr, rgb=False, multiscale=False, name='APR', translate=[0, 0, 2048]))
    layers.append(Image(data=aprarr, rgb=False, multiscale=False, name='APR', translate=[0, 2048, 0]))
    layers.append(Image(data=aprarr, rgb=False, multiscale=False, name='APR', translate=[0, 2048, 2048]))
    # Display APR
    viewer = display_layers(layers)


    # Segmentation display exmaple
    mask = parts>1500
    display_segmentation(apr, parts, mask)
