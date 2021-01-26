import pyapr
import numpy as np
from skimage.io import imsave

# APR file to segment
fpath_apr = r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/2P_mouse_re0.2.apr'
threshold = 5000

# Instantiate APR and particle objects
apr = pyapr.APR()
parts = pyapr.ShortParticles()  # input particles can be float32 or uint16

# Read from APR file
pyapr.io.read(fpath_apr, apr, parts)

# Create segmentation mask
mask = np.array(parts, copy=False)

# Iterate over APR structure to obtain level
apr_it = apr.iterator()
part_position = []
part_level = []
part_intensity = []
for level in range(apr_it.level_min(), apr_it.level_max()+1):
    for z in range(apr_it.z_num(level)):
        for x in range(apr_it.x_num(level)):
            for idx in range(apr_it.begin(level, z, x), apr_it.end()):
                y = apr_it.y(idx)

                # Save data in arrays
                part_position.append([z, y, x])
                part_level.append(level)
                part_intensity.append(parts[idx])

                # Construct mask
                if level < apr_it.level_max() - 1:
                    mask[idx] = 0
                if parts[idx] < threshold:
                    mask[idx] = 0

mask[mask!=0] = 1

# Compute closing with increasing radius
for r in range(10):
    parts_closing = pyapr.numerics.transform.closing(apr, parts, binary=True, radius=r, inplace=False)
    pc_recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts_closing), copy=False)
    imsave(r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/closing_tests/closing_radius{}.tif'.format(r), pc_recon[12], check_contrast=False)

# Compute several closing with radius 1
for n in range(10):
    if n>0:
        pyapr.numerics.transform.closing(apr, parts, binary=True, radius=1, inplace=True)
    pc_recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)
    imsave(r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/closing_tests/closing_radius1_n{}.tif'.format(n), pc_recon[12], check_contrast=False)

# Compute dilation with increasing radius
for r in range(10):
    parts_closing = parts.copy()
    pyapr.numerics.transform.dilation(apr, parts_closing, binary=True, radius=r)
    pc_recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts_closing), copy=False)
    imsave(r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/dilation_tests/dilation_radius{}.tif'.format(r), pc_recon[12], check_contrast=False)

# Compute several dilation with radius 1
parts_closing = parts.copy()
for n in range(10):
    if n>0:
        pyapr.numerics.transform.dilation(apr, parts, binary=True, radius=1)
    pc_recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, parts), copy=False)
    imsave(r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/dilation_tests/dilation_radius1_n{}.tif'.format(n), pc_recon[12], check_contrast=False)