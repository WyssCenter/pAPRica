import pyapr
import numpy as np

def gaussian_blur(apr, parts, sigma=1.5, size=11):
    stencil = pyapr.numerics.get_gaussian_stencil(size, sigma, 3, True)
    output = pyapr.FloatParticles()
    pyapr.numerics.filter.convolve_pencil(apr, parts, output, stencil, use_stencil_downsample=True,
                                          normalize_stencil=True, use_reflective_boundary=True)
    return output

par = pyapr.APRParameters()
par.rel_error = 0.2
par.gradient_smoothing = 2
par.dx = 1
par.dy = 1
par.dz = 1
par.Ip_th = 30000
par.sigma_th = 10.0
par.grad_th = 10.0
nz = 10

u = (np.random.rand(nz, 2048, 2048)*2**15).astype('uint16')
apr, parts = pyapr.converter.get_apr(u, params=par)

u = pyapr.numerics.reconstruction.reconstruct_constant(apr, parts)
gauss = gaussian_blur(apr, parts)
pyapr.viewer.parts_viewer(apr, gauss)

