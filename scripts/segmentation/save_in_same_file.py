"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pyapr
import os

path_intensity = '/home/jules/Desktop/apr_pipelines/data/apr/1_3.apr'
path_cc = '/home/jules/Desktop/apr_pipelines/data/segmentation/1_3_segmentation.apr'
path_output = '/home/jules/Desktop'

# Read APR data
apr = pyapr.APR()
parts = pyapr.ShortParticles()
pyapr.io.read(path_intensity, apr, parts)

# Save APR file
pyapr.io.write(os.path.join(path_output, 'test.apr'), apr, parts)
pyapr.io.write(os.path.join(path_output, 'intensity.apr'), apr, parts)

# Read CC data
apr = pyapr.APR()
cc = pyapr.LongParticles()
pyapr.io.read(path_cc, apr, cc)
pyapr.io.write(os.path.join(path_output, 'cc.apr'), apr, cc)

# Save CC data on intensity data
aprfile = pyapr.io.APRFile()
aprfile.set_read_write_tree(False)
aprfile.open(os.path.join(path_output, 'test.apr'), 'READWRITE')
aprfile.write_particles('segmentation cc', cc, t=0)
aprfile.close()