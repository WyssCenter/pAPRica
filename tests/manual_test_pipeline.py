"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from time import time

import paprica
import os
import numpy as np
import pyapr


def compute_features(apr, parts):
    gauss = paprica.segmenter.gaussian_blur(apr, parts, sigma=1.5, size=11)
    print('Gaussian computed.')

    # Compute gradient magnitude (central finite differences)
    grad = paprica.segmenter.compute_gradmag(apr, gauss)
    print('Gradient magnitude computed.')
    # Compute lvl for each particle
    lvl = paprica.segmenter.particle_levels(apr)
    print('Particle level computed.')
    # Compute difference of Gaussian
    dog = paprica.segmenter.gaussian_blur(apr, parts, sigma=3, size=22) - gauss
    print('DOG computed.')
    lapl_of_gaussian = paprica.segmenter.compute_laplacian(apr, gauss)
    print('Laplacian of Gaussian computed.')

    # Aggregate filters in a feature array
    f = np.vstack((np.array(parts, copy=True),
                   lvl,
                   gauss,
                   grad,
                   lapl_of_gaussian,
                   dog
                   )).T

    return f


def get_cc_from_features(apr, parts_pred):

    # Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
    parts_cells = (parts_pred == 1)

    # Use opening to separate touching cells
    pyapr.morphology.opening(apr, parts_cells, radius=1, binary=True, inplace=True)

    # Apply connected component
    cc = pyapr.LongParticles()
    pyapr.measure.connected_component(apr, parts_cells, cc)

    # Remove small and large objects
    cc = pyapr.numerics.transform.remove_small_objects(apr, cc, 128)
    cc = pyapr.numerics.transform.remove_large_objects(apr, cc, 1024)

    return cc


def main():
    # Parameters
    path = r'data/apr/'
    path_classifier=r'../data/random_forest_n100.joblib'
    n = 1

    # Parse data
    t_ini = time()
    tiles = paprica.parser.tileParser(path, frame_size=512, ftype='apr')
    t = time()

    # Stitch
    stitcher = paprica.stitcher.tileStitcher(tiles, overlap_h=25, overlap_v=25)

    t = time()
    stitcher.compute_registration()
    print('Elapsed time old registration: {} s.'.format((time()-t)/n))
    t = time()
    stitcher.compute_registration_fast()
    print('Elapsed time new registration on RAM: {} s.'.format((time()-t)/n))
    t = time()
    stitcher.compute_registration_fast(on_disk=True)
    print('Elapsed time new registration on disk: {} s.'.format((time()-t)/n))

    stitcher.save_database(os.path.join(path, 'registration_results.csv'))

    # Segment and extract objects across the whole volume
    trainer = paprica.tileTrainer(tiles[0],
                                  func_to_compute_features=compute_features,
                                  func_to_get_cc=get_cc_from_features)
    trainer.manually_annotate()
    trainer.train_classifier(n_estimators=100)
    segmenter = paprica.segmenter.multitileSegmenter(tiles, stitcher.database, clf=trainer.clf,
                                                     func_to_compute_features=compute_features,
                                                     func_to_get_cc=get_cc_from_features)
    segmenter.compute_multitile_segmentation(save_cc=True)

    # Display result
    viewer = paprica.viewer.tileViewer(tiles, stitcher.database, segmentation=True, cells=segmenter.cells)
    viewer.display_all_tiles(pyramidal=True, downsample=1, contrast_limits=[0, 3000])


if __name__ == '__main__':
  main()