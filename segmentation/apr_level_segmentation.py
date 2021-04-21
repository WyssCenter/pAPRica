"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

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

# Display APR
# pyapr.viewer.parts_viewer(apr, parts)

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
pyapr.numerics.transform.closing(apr, parts, binary=True, radius=1, inplace=True)
pyapr.viewer.parts_viewer(apr, parts)

# Perform connected component labeling
comp_labels = pyapr.ShortParticles()
pyapr.numerics.segmentation.connected_component(apr, parts, comp_labels)

pyapr.viewer.parts_viewer(apr, comp_labels)

# Save segmentation as tif
# pc_recon = np.array(pyapr.numerics.reconstruction.recon_pc(apr, comp_labels), copy=False)
# imsave(r'/media/sf_shared_folder_virtualbox/mouse_2P/data1/lmap_from_apr.tif', pc_recon, check_contrast=False)
