'''
This script takes the registered atlas and upsamples it to the original data.
'''

from skimage.io import imread
import pyapr
import numpy as np
import matplotlib.pyplot as plt
import napari

# Parameters
atlas_path = r'/mnt/Data/wholebrain/brainreg_output_autofluo_clahe_crop/registered_atlas.tiff'


atlas = imread(atlas_path)

apr = pyapr.APR()
parts = pyapr.FloatParticles()
par = pyapr.APRParameters()

par.rel_error = 0.00001
par.gradient_smoothing = 0
par.grad_th = 0.001
par.sigma_th = 0.001
par.Ip_th = 0

apr, parts = pyapr.converter.get_apr(atlas.astype('float32'), params=par)
# pyapr.viewer.parts_viewer(apr, parts)
toto = pyapr.numerics.reconstruction.reconstruct_constant(apr, parts)
print(apr.computational_ratio())
print(np.sum(toto!=atlas)/toto.size*100)
i = 250
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
ax[0].imshow(toto[i]!=atlas[i])
ax[1].imshow(atlas[i])
ax[1].set_title('ATLAS')
ax[2].imshow(toto[i])
ax[2].set_title('APR')

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(atlas)