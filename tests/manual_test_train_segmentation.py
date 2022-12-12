"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import paprica
from paprica.segmenter import compute_laplacian, compute_gradmag, gaussian_blur, particle_levels
import pyapr
import numpy as np

def compute_features(apr, parts):
    gauss = gaussian_blur(apr, parts, sigma=1.5, size=11)
    print('Gaussian computed.')

    # Compute gradient magnitude (central finite differences)
    grad = compute_gradmag(apr, gauss)
    print('Gradient magnitude computed.')
    # Compute lvl for each particle
    lvl = particle_levels(apr)
    print('Particle level computed.')
    # Compute difference of Gaussian
    dog = gaussian_blur(apr, parts, sigma=3, size=22) - gauss
    print('DOG computed.')
    lapl_of_gaussian = compute_laplacian(apr, gauss)
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

    # Remove small objects
    # cc = pyapr.numerics.transform.remove_small_objects(apr, cc, 128)

    return cc


def main():
    # Parameters
    path = 'data/apr'

    # We load a tile
    tiles = paprica.parser.tileParser(path, frame_size=512, overlap=128, ftype='apr')
    tile = tiles[2]
    tile.load_tile()

    # We create the trainer object and then manually annotate the dataset
    trainer = paprica.segmenter.tileTrainer(tile, compute_features)
    trainer.manually_annotate(use_sparse_labels=True)
    # trainer.save_labels()
    # trainer.load_labels()

    # We then train the classifier and apply it to the whole training tile
    trainer.train_classifier()
    trainer.display_training_annotations(contrast_limits=[0, 10000])
    trainer.segment_training_tile(bg_label=2)
    # trainer.apply_on_tile(tiles[7], bg_label=2, func_to_get_cc=get_cc_from_features)
    # trainer.apply_on_tile(tiles[7], bg_label=2)


if __name__ == '__main__':
  main()