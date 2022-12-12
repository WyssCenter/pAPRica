"""
This is a script that shows how to segment data.


By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import paprica
import pyapr
import numpy as np
from time import time

def compute_features(apr, parts):
    t = time()
    gauss = paprica.segmenter.gaussian_blur(apr, parts, sigma=1.5, size=11)
    print('Gaussian computed.')

    # Compute gradient magnitude (central finite differences)
    grad = paprica.segmenter.compute_gradmag(apr, gauss)
    print('Gradient magnitude computed.')
    # Compute local standard deviation around each particle
    # local_std = compute_std(apr, parts, size=5)
    # print('STD computed.')
    # Compute lvl for each particle
    lvl = paprica.segmenter.particle_levels(apr)
    print('Particle level computed.')
    # Compute difference of Gaussian
    dog = paprica.segmenter.gaussian_blur(apr, parts, sigma=3, size=22) - gauss
    print('DOG computed.')
    lapl_of_gaussian = paprica.segmenter.compute_laplacian(apr, gauss)
    print('Laplacian of Gaussian computed.')

    print('Features computation took {} s.'.format(time()-t))

    # Aggregate filters in a feature array
    f = np.vstack((np.array(parts, copy=True),
                   lvl,
                   gauss,
                   grad,
                   lapl_of_gaussian,
                   dog
                   )).T

    return f

# First we define the path where the data is located
path = '/home/user/folder_containing_data'

# If you don't have any data to try on, you can run the 'example_create_synthetic_dataset.py' script

# We then parse this data using the parser
tiles = paprica.tileParser(path=path, frame_size=2048, ftype='apr')

# Here we will manually train the segmenter on the first tile
trainer = paprica.tileTrainer(tiles[0, 0], func_to_compute_features=compute_features)
trainer.manually_annotate()
# We can save manual labels and add more labels later on
trainer.save_labels()
trainer.add_annotations()
# Now, let's train the classifier
trainer.train_classifier()
# We can apply it on the first tile to see how it performs
trainer.segment_training_tile()

# Now we can segment all tiles using the trained classifier
segmenter = paprica.tileSegmenter.from_trainer(trainer)
for tile in tiles:
    segmenter.compute_segmentation(tile)

# Note that it is possible to perform the stitching and the segmentation at the same time, optimizing IO operations:
stitcher = paprica.tileStitcher(tiles, overlap_h=20, overlap_v=20)
stitcher.activate_segmentation(segmenter)
stitcher.compute_registration()
