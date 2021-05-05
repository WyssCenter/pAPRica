"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

from glob import glob
import os
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import pyapr
from skimage.registration import phase_cross_correlation
from joblib import load
from scipy.signal import correlate
from time import time

class tileLoader():
    """
    Class to load each tile and neighboring tiles data, perform the registration and segmentation.
    A tileGraph object must be initialized and passed for computing the registration.

    Tile post processing is done on APR data, so if the input data is tiff it is first converted.
    """
    def __init__(self, tile):
        # TODO: here the data of each tile is passed as a dictionary. Maybe there is a less redundant or cleaner way
        # to achieve this.
        self.path = tile['path']
        self.row = tile['row']
        self.col = tile['col']
        self.type = tile['type']
        self.neighbors = tile['neighbors']
        self.neighbors_path = tile['neighbors_path']
        self.overlap = tile['overlap']
        self.frame_size = tile['frame_size']
        self.mask = False
        self.threshold = None

        # Load tile data and neighbors data.
        self.data = self._load_tile(self.path)
        self.data_neighbors = self._load_neighbors()

        # If data is not APR then convert it
        if self.type != 'apr':
            self._convert_to_apr()

        # Initialize attributes for segmentation
        self.path_classifier = None
        self.f_names = None

    def activate_mask(self, threshold):
        """
        Activate the masked cross-correlation for the displacement estimation. Pixels above threshold are
        not taken into account.

        Parameters
        ----------
        threshold: (int) threshold for the cross-correlation mask as a percentage of pixel to keep (e.g. 95 will
                    create a mask removing the 5% brightest pixels).

        """
        self.mask = True
        self.threshold = threshold

    def deactivate_mask(self):
        """
        Deactivate the masked cross-correlation and uses a classical cross correlation.

        """
        self.mask = False
        self.threshold = None

    def compute_registration(self, tgraph):
        """
        Compute the pair-wise registration for a given tile with all its neighbors (EAST and SOUTH to avoid
        the redundancy).

        Parameters
        ----------
        tgraph: (tileGraph object) stores pair-wise registration to further perform the global optimization.

        """
        for v, coords in zip(self.data_neighbors, self.neighbors):
            if self.row == coords[0] and self.col < coords[1]:
                # EAST
                reg, rel = self._compute_east_registration(v)

            elif self.col == coords[1] and self.row < coords[0]:
                # SOUTH
                reg, rel = self._compute_south_registration(v)

            else:
                raise TypeError('Error: couldn''t determine registration to perform.')

            tgraph.cgraph_from.append(np.ravel_multi_index([self.row, self.col], dims=(tgraph.nrow, tgraph.ncol)))
            tgraph.cgraph_to.append(np.ravel_multi_index([coords[0], coords[1]], dims=(tgraph.nrow, tgraph.ncol)))
            # H=x, V=y, D=z
            tgraph.dH.append(reg[2])
            tgraph.dV.append(reg[1])
            tgraph.dD.append(reg[0])
            tgraph.relia_H.append(rel[2])
            tgraph.relia_V.append(rel[1])
            tgraph.relia_D.append(rel[0])

    # def compute_segmentation(self, path_classifier, func_to_compute_features):
    #     """
    #     Compute the segmentation (iLastik like) on the tile. Before attempting this, a model needs to be trained and
    #     saved.
    #
    #     Parameters
    #     ----------
    #     path_classifier: (str) path to the trained model for the classification.
    #
    #     """
    #
    #     if path_classifier is None:
    #         raise ValueError('Error: no classifier path was given.')
    #
    #     # Load classifier
    #     clf = load(path_classifier)
    #
    #     # Compute features
    #     # TODO: is there a way to adapt the computed features to the ones used to train the classifier?
    #     f = self._compute_features()
    #     apr = self.data[0]
    #
    #     # Predict particle type (cell, membrane or brackground) for each cell with the trained model
    #     parts_pred = self._predict_on_APR(clf, f)
    #     # Create a mask from particle classified as cells (cell=0, background=1, membrane=2)
    #     parts_cells = (parts_pred == 0)
    #     # Remove small holes to get the misclassified nuclei
    #     parts_cells = pyapr.numerics.transform.remove_small_holes(apr, parts_cells, min_volume=500)
    #     # Opening to better separate touching cells
    #     pyapr.numerics.transform.opening(apr, parts_cells, radius=1, inplace=True)
    #     # Apply connected component
    #     cc = pyapr.ShortParticles()
    #     pyapr.numerics.segmentation.connected_component(apr, parts_cells, cc)
    #     # Remove small objects
    #     pyapr.numerics.transform.remove_small_objects(apr, cc, min_volume=200)
    #     # Save segmentation results
    #     pyapr.io.write(os.path.join(self.path, 'segmentation.apr'), apr, cc)

    def init_segmentation(self, path_classifier, func_to_compute_features, func_to_get_cc):

        # Load classifier
        self.clf = load(path_classifier)

        # Store function to compute features
        self.func_to_compute_features = func_to_compute_features

        # Store post processing steps
        self.func_to_get_cc = func_to_get_cc

    def compute_segmentation(self, verbose=False):

        apr = self.data[0]
        parts = self.data[1]

        if verbose:
            t = time()
            print('Computing features on AP')
        f = self.func_to_compute_features(apr, parts)

        parts_pred = self._predict_on_APR_block(f, verbose=verbose)

        if verbose:
            # Display inference info
            print('\n\n****** INFERENCE RESULTS ******\n')
            print(
                '{} cell particles ({:0.2f}%)'.format(np.sum(parts_pred == 0),
                                                      np.sum(parts_pred == 0) / len(parts_pred) * 100))
            print('{} background particles ({:0.2f}%)'.format(np.sum(parts_pred == 1),
                                                              np.sum(parts_pred == 1) / len(parts_pred) * 100))
            print('{} membrane particles ({:0.2f}%)'.format(np.sum(parts_pred == 2),
                                                            np.sum(parts_pred == 2) / len(parts_pred) * 100))

        cc = self.func_to_get_cc(apr, parts_pred)

        pyapr.io.write(self.path[:-4] + '_segmentation.apr', apr, cc)

    def _load_tile(self, path):
        """
        Load the current tile.
        """
        if self.type == 'tiff2D':
            files = glob(os.path.join(path, '*.tif'))
            im = imread(files[0])
            u = np.zeros((len(files), *im.shape))
            u[0] = im
            files.pop(0)
            for i, file in enumerate(files):
                u[i+1] = imread(file)
        elif self.type == 'tiff3D':
            u = imread(path)
        elif self.type == 'apr':
            apr = pyapr.APR()
            parts = pyapr.ShortParticles()
            pyapr.io.read(path, apr, parts)
            u = (apr, parts)
        else:
            raise TypeError('Error: image type {} not supported.'.format(self.type))
        return u

    def _load_neighbors(self):
        """
        Load the current tile neighbors.
        """
        u = []
        for neighbor in self.neighbors_path:
            u.append(self._load_tile(neighbor))
        return u

    def _convert_to_apr(self):
        """
        Converts input tile from pixel data to APR.
        """
        # TODO: have an automatic way to set the parameters.

        # Parameters are hardcoded for now
        par = pyapr.APRParameters()
        par.auto_parameters = False  # really heuristic and not working
        par.sigma_th = 26.0
        par.grad_th = 3.0
        par.Ip_th = 253.0
        par.rel_error = 0.2
        par.gradient_smoothing = 2

        # Convert data to APR
        self.data = pyapr.converter.get_apr(image=self.data, params=par, verbose=False)

        # Convert neighbors to APR
        data_apr = []
        for data in self.data_neighbors:
            data_apr.append(pyapr.converter.get_apr(image=data, params=par, verbose=False))
        self.data_neighbors = data_apr

    def _compute_features(self):
        """
        Returns the features computed on APR data for using with the classifier to produce the segmentation.
        """
        apr = self.data[0]
        parts = self.data[1]

        # Compute gradient along a dimension (Sobel filter). dimension can be 0, 1 or 2
        grad_x, grad_y, grad_z = self._compute_gradients(apr, parts)

        # Compute gradient magnitude (central finite differences)
        grad = self._compute_gradmag(apr, parts)

        # Compute local standard deviation around each particle
        local_std = self._compute_std(apr, parts, size=5)

        # Compute lvl for each particle
        lvl = self._particle_levels(apr, normalize=True)

        # Compute difference of Gaussian
        dog = self._gaussian_blur(apr, parts, sigma=3, size=22) - self._gaussian_blur(apr, parts, sigma=1.5, size=11)

        # Aggregate filters in a feature array
        f = np.vstack((np.array(parts, copy=True),
                       lvl,
                       grad_x,
                       grad_y,
                       grad_z,
                       grad,
                       local_std,
                       dog
                       )).T
        if self.f_names is None:
            self.f_names = ['Intensity',
                            'lvl',
                            'grad_x',
                            'grad_y',
                            'grad_z',
                            'grad_mag',
                            'local_std',
                            'dog'
                            ]
        return f

    def _compute_gradients(self, apr, parts, sobel=True):
        """
        Returns the gradients [dz, dx, dy] of APR data.
        """
        par = apr.get_parameters()
        dx = pyapr.FloatParticles()
        dy = pyapr.FloatParticles()
        dz = pyapr.FloatParticles()

        pyapr.numerics.gradient(apr, parts, dz, dimension=2, delta=par.dz, sobel=sobel)
        pyapr.numerics.gradient(apr, parts, dx, dimension=1, delta=par.dx, sobel=sobel)
        pyapr.numerics.gradient(apr, parts, dy, dimension=0, delta=par.dy, sobel=sobel)
        return dz, dx, dy

    def _compute_laplacian(self, apr, parts, sobel=True):
        """
        Returns the Laplacian of APR data.
        """
        # TODO: merge this with compute gradient to avoid computing it twice.
        par = apr.get_parameters()
        dz, dx, dy = self._compute_gradients(apr, parts, sobel)
        dx2 = pyapr.FloatParticles()
        dy2 = pyapr.FloatParticles()
        dz2 = pyapr.FloatParticles()
        pyapr.numerics.gradient(apr, dz, dz2, dimension=2, delta=par.dz, sobel=sobel)
        pyapr.numerics.gradient(apr, dx, dx2, dimension=1, delta=par.dx, sobel=sobel)
        pyapr.numerics.gradient(apr, dy, dy2, dimension=0, delta=par.dy, sobel=sobel)
        return dz + dx + dy

    def _compute_gradmag(self, apr, parts, sobel=True):
        """
        Returns the gradient magnitude of APR data.
        """
        # TODO: idem laplacian, this can probably be optimized.
        par = apr.get_parameters()
        gradmag = pyapr.FloatParticles()
        pyapr.numerics.gradient_magnitude(apr, parts, gradmag, deltas=(par.dz, par.dx, par.dy), sobel=True)
        return gradmag

    def _gaussian_blur(self, apr, parts, sigma=1.5, size=11):
        """
        Returns a gaussian filtered APR data.
        """
        stencil = pyapr.numerics.get_gaussian_stencil(size, sigma, 3, True)
        output = pyapr.FloatParticles()
        pyapr.numerics.filter.convolve_pencil(apr, parts, output, stencil, use_stencil_downsample=True,
                                              normalize_stencil=True, use_reflective_boundary=True)
        return output

    def _particle_levels(self, apr, normalize=True):
        """
        Returns the particle level of APR data.
        """
        lvls = pyapr.ShortParticles(apr.total_number_particles())
        lvls.fill_with_levels(apr)
        if normalize:
            lvls *= (1 / apr.level_max())
        return lvls

    def _compute_std(self, apr, parts, size=5):
        """
        Returns the local std of APR data.
        """
        dims = apr.org_dims()
        box_size = [size if d >= size else 1 for d in dims]
        locstd = pyapr.FloatParticles()
        pyapr.numerics.local_std(apr, parts, locstd, size=box_size)
        return locstd

    def _predict_on_APR(self, clf, x):
        """
        Predict particle class with the trained classifier clf on the precomputed features f.
        """
        # Predict on numpy array
        y_pred = clf.predict(x)

        # Transform numpy array to ParticleData
        parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

        return parts_pred

    def _predict_on_APR_block(self, x, n_parts=1e7, verbose=False):
        """
        Predict particle class with the trained classifier clf on the precomputed features f using a
        blocked strategy to avoid memory segfault.
        """
        # Predict on numpy array by block to avoid memory issues
        if verbose:
            t = time()

        y_pred = np.empty((x.shape[0]))
        n_block = int(np.ceil(x.shape[0] / n_parts))
        if int(n_parts) != n_parts:
            raise ValueError('Error: n_parts must be an int.')
        n_parts = int(n_parts)

        self.clf[1].set_params(n_jobs=-1)
        for i in range(n_block):
            y_pred[i * n_parts:min((i + 1) * n_parts, x.shape[0])] = self.clf.predict(
                x[i * n_parts:min((i + 1) * n_parts, x.shape[0])])

        if verbose:
            print('Blocked prediction took {} s.\n'.format(time() - t))

        # Transform numpy array to ParticleData
        parts_pred = pyapr.ShortParticles(y_pred.astype('uint16'))

        return parts_pred

    def _get_proj_shifts(self, proj1, proj2, upsample_factor=1):
        """
        This function computes shifts from max-projections on overlapping areas. It uses the phase cross-correlation
        to compute the shifts.

        Parameters
        ----------
        proj1: (list of arrays) max-projections for tile 1
        proj2: (list of arrays) max-projections for tile 2

        Returns
        -------
        shifts in (x, y, z) and error measure (0=reliable, 1=not reliable)

        """
        # Compute phase cross-correlation to extract shifts
        dzy, error_zy, _ = phase_cross_correlation(proj1[0], proj2[0],
                                                   return_error=True, upsample_factor=upsample_factor)
        dzx, error_zx, _ = phase_cross_correlation(proj1[1], proj2[1],
                                                   return_error=True, upsample_factor=upsample_factor)
        dyx, error_yx, _ = phase_cross_correlation(proj1[2], proj2[2],
                                                   return_error=True, upsample_factor=upsample_factor)

        # Keep only the most reliable registration
        # D/z
        if error_zx < error_zy:
            dz = dzx[0]
            rz = error_zx
        else:
            dz = dzy[0]
            rz = error_zy

        # H/x
        if error_zx < error_yx:
            dx = dzx[1]
            rx = error_zx
        else:
            dx = dyx[1]
            rx = error_yx

        # V/y
        if error_yx < error_zy:
            dy = dyx[0]
            ry = error_yx
        else:
            dy = dzy[1]
            ry = error_zy

        # for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('dx={}, dy={}, dz={}'.format(dx, dy, dz))
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)
        #
        # if self.row==0 and self.col==0:
        #     print('ok')

        return np.array([dz, dy, dx]), np.array([rz, ry, rx])

    def _get_masked_proj_shifts(self, proj1, proj2, upsample_factor=1):
        """
        This function computes shifts from max-projections on overlapping areas with mask on brightest area.
        It uses the phase cross-correlation to compute the shifts.

        Parameters
        ----------
        proj1: (list of arrays) max-projections for tile 1
        proj2: (list of arrays) max-projections for tile 2

        Returns
        -------
        shifts in (x, y, z) and error measure (0=reliable, 1=not reliable)

        """
        # Compute mask to discard very bright area that are likely bubbles or artefacts
        mask_ref = []
        mask_move = []
        for i in range(3):
            vmax = np.percentile(proj1[i], self.threshold)
            mask_ref.append(proj1[i] < vmax)
            vmax = np.percentile(proj2[i], self.threshold)
            mask_move.append(proj2[i] < vmax)

        # Compute phase cross-correlation to extract shifts
        dzy = phase_cross_correlation(proj1[0], proj2[0],
                                      return_error=True, upsample_factor=upsample_factor,
                                      reference_mask=mask_ref[0], moving_mask=mask_move[0])
        error_zy = self._get_registration_error(proj1[0], proj2[0])
        dzx = phase_cross_correlation(proj1[1], proj2[1],
                                      return_error=True, upsample_factor=upsample_factor,
                                      reference_mask=mask_ref[1], moving_mask=mask_move[1])
        error_zx = self._get_registration_error(proj1[1], proj2[1])
        dyx = phase_cross_correlation(proj1[2], proj2[2],
                                      return_error=True, upsample_factor=upsample_factor,
                                      reference_mask=mask_ref[2], moving_mask=mask_move[2])
        error_yx = self._get_registration_error(proj1[2], proj2[2])

        # Keep only the most reliable registration
        # D/z
        if error_zx < error_zy:
            dz = dzx[0]
            rz = error_zx
        else:
            dz = dzy[0]
            rz = error_zy

        # H/x
        if error_zx < error_yx:
            dx = dzx[1]
            rx = error_zx
        else:
            dx = dyx[1]
            rx = error_yx

        # V/y
        if error_yx < error_zy:
            dy = dyx[0]
            ry = error_yx
        else:
            dy = dzy[1]
            ry = error_zy

        # for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('dx={}, dy={}, dz={}'.format(dx, dy, dz))
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)
        #
        # if self.row==0 and self.col==0:
        #     print('ok')

        return np.array([dz, dy, dx]), np.array([rz, ry, rx])

    def _get_registration_error(self, proj1, proj2):
        return np.sqrt(1-correlate(proj1, proj2).max()**2/(np.sum(proj1**2)*np.sum(proj2**2)))

    def _get_max_proj_apr(self, apr, parts, patch, plot=False):
        """
        Get the maximum projection from 3D APR data.
        """
        proj = []
        for d in range(3):
            # dim=0: project along Y to produce a ZY plane
            # dim=1: project along X to produce a ZX plane
            # dim=2: project along Z to produce an YX plane
            proj.append(pyapr.numerics.transform.projection.maximum_projection_patch(apr, parts, dim=d, patch=patch))

        if plot:
            fig, ax = plt.subplots(1, 3)
            for i, title in enumerate(['ZY', 'ZX', 'YX']):
                ax[i].imshow(proj[i], cmap='gray')
                ax[i].set_title(title)

        return proj[0], proj[1], proj[2]

    def _compute_east_registration(self, v):
        """
        Compute the registration between the current tile and its eastern neighbor.
        """
        apr_1, parts_1 = self.data
        apr_2, parts_2 = v

        patch = pyapr.ReconPatch()
        patch.y_begin = self.frame_size - self.overlap
        proj_zy1, proj_zx1, proj_yx1 = self._get_max_proj_apr(apr_1, parts_1, patch, plot=False)

        patch = pyapr.ReconPatch()
        patch.y_end = self.overlap
        proj_zy2, proj_zx2, proj_yx2 = self._get_max_proj_apr(apr_2, parts_2, patch, plot=False)

        # proj1, proj2 = [proj_zy1, proj_zx1, proj_yx1], [proj_zy2, proj_zx2, proj_yx2]
        # for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('EAST')
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)

        # if self.row==0 and self.col==1:
        #     print('ok')

        if self.mask:
            return self._get_masked_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                         [proj_zy2, proj_zx2, proj_yx2])
        else:
            return self._get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                         [proj_zy2, proj_zx2, proj_yx2])

    def _compute_south_registration(self, v):
        """
        Compute the registration between the current tile and its southern neighbor.
        """
        apr_1, parts_1 = self.data
        apr_2, parts_2 = v

        patch = pyapr.ReconPatch()
        patch.x_begin = self.frame_size - self.overlap
        proj_zy1, proj_zx1, proj_yx1 = self._get_max_proj_apr(apr_1, parts_1, patch, plot=False)

        patch = pyapr.ReconPatch()
        patch.x_end = self.overlap
        proj_zy2, proj_zx2, proj_yx2 = self._get_max_proj_apr(apr_2, parts_2, patch, plot=False)

        # proj1, proj2 = [proj_zy1, proj_zx1, proj_yx1], [proj_zy2, proj_zx2, proj_yx2]
        # for i, title in enumerate(['ZY', 'ZX', 'YX']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('EAST')
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)

        # if self.row==0 and self.col==1:
        #     print('ok')

        if self.mask:
            return self._get_masked_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                         [proj_zy2, proj_zx2, proj_yx2])
        else:
            return self._get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                         [proj_zy2, proj_zx2, proj_yx2])