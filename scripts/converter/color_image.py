"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from skimage.color import rgb2hsv, hsv2rgb
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
import pyapr
import numpy as np
import os

path = '/home/jules/Desktop/jules.jpg'

u = imread(path)
if u.shape[2]>3:
    u = u[:,:,:3]

u_hsv = rgb2hsv(u)
u_hsv_16bits = np.empty_like(u_hsv, dtype='uint16')
for i in range(3):
    u_hsv_16bits[:,:,i] = rescale_intensity(u_hsv[:,:,i], in_range=(0, 1), out_range='uint16').astype('uint16')

par = pyapr.APRParameters()
par.Ip_th = u_hsv_16bits[:,:,2].mean()-1.8*u_hsv_16bits[:,:,2].std()
par.auto_parameters = True

apr, parts = pyapr.converter.get_apr(u_hsv_16bits[:,:,2], rel_error=0.8, params=par)
V_new = pyapr.numerics.reconstruction.reconstruct_constant(apr, parts).squeeze()
V_new = rescale_intensity(V_new, out_range='float')

p = pyapr.ShortParticles()
p.sample_image(apr, u_hsv_16bits[:,:,1])
S_new = pyapr.numerics.reconstruction.reconstruct_constant(apr, p).squeeze()
S_new = rescale_intensity(S_new, out_range='float')

p = pyapr.ShortParticles()
p.sample_image(apr, u_hsv_16bits[:,:,0])
H_new = pyapr.numerics.reconstruction.reconstruct_constant(apr, p).squeeze()
H_new = rescale_intensity(H_new, out_range='float')

u_apr = hsv2rgb(np.dstack((H_new, S_new, V_new)))
u_apr = rescale_intensity(u_apr, out_range='uint8').astype('uint8')

imsave(path[:-4]+ '_apr.png', u_apr)
