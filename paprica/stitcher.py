"""
Submodule containing classes and functions relative to **stitching**.

With this submodule the user can stitch a previously parsed dataset, typically the autofluorescence channel:

>>> import paprica
>>> tiles_autofluo = paprica.parser.tileParser(path_to_autofluo, frame_size=1024, overlap=25)
>>> stitcher = paprica.stitcher.tileStitcher(tiles_autofluo)
>>> stitcher.compute_registration_fast()

Others channel can then easily stitched using the previous one as reference:

>>> tiles_signal = paprica.parser.tileParser(path_to_data, frame_size=1024, overlap=25)
>>> stitcher_channel = paprica.stitcher.channelStitcher(stitcher, tiles_autofluo, tiles_signal)
>>> stitcher_channel.compute_rigid_registration()

Doing that each tile in the second data set will be registered to the corresponding autofluorescence tile and
then their spatial position will be adjusted.

WARNING: when stitching, the expected overlap must be HIGHER than the real one. To enforce this, a margin of 20% is
automatically taken (this margin can be set lower by the user for speed improvement). In order to get the best stitching
quality it requires to have a good estimate of the overlap, hence why the full volume is not considered.

This submodule also contains a class for merging and reconstructing the data. It was intended to be used at lower
resolution for atlasing. The generated data can quickly become out of hands, use with caution!

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import os
import warnings
from pathlib import Path

import cv2 as cv
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyapr
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from skimage.registration import phase_cross_correlation
from scipy.signal import correlate
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from skimage.color import label2rgb, hsv2rgb
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import gaussian
from skimage.metrics import normalized_root_mse
from skimage.transform import warp, AffineTransform, downscale_local_mean
from tqdm import tqdm
import napari


def max_sum_over_single_max(reference_image, moving_image, d):
    """
    This function is a reliability metric which works well for sparse data. It computes the 99 percentile of the sum
    of reference and shifted image divided by twice the 99 percentile of the reference image.

    Parameters
    ----------
    reference_image: ndarray
        2D array of the reference image
    moving_image: ndarray
        2D array of the moving image image
    d: array_like
        registration parameters

    Returns
    -------
    e: float
        error estimation of the registration. The lower e, the more reliable the registration.
    """

    shifted_image = warp(moving_image, AffineTransform(translation=[d[1], d[0]]), mode='wrap', preserve_range=True)

    e = (2*np.percentile(reference_image, 99))/np.percentile(reference_image+shifted_image, 99)

    return e


def mse(reference_image, moving_image, d):
    """
    Normalized root mean square error.

    Parameters
    ----------
    reference_image: ndarray
        2D array of the reference image
    moving_image: ndarray
        2D array of the moving image image
    d: array_like
        registration parameters

    Returns
    -------
    _: float
        error estimation of the registration. The lower e, the more reliable the registration.

    """

    shifted_image = warp(moving_image, AffineTransform(translation=[d[1], d[0]]), mode='wrap',
                         preserve_range=True)

    return normalized_root_mse(reference_image, shifted_image, normalization='mean')


def phase_cross_correlation(reference_image,
                            moving_image,
                            upsample_factor=1,
                            return_error=True):
    """
    Phase cross correlation. Because skimage function compute the NORMAL cross correlation to estimate the shift I
    modified it to compute the TRUE phase cross correlation, as per the standard definition.

    Parameters
    ----------
    reference_image : array
        Reference image.
    moving_image : array
        Image to register. Must be same dimensionality as
        ``reference_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).
        Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    return_error : bool, optional
        Returns error and phase difference if on, otherwise only
        shifts are returned. Has noeffect if any of ``reference_mask`` or
        ``moving_mask`` is not None. In this case only shifts is returned.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between
        ``reference_image`` and ``moving_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).
    """

    # images must be the same shape
    if reference_image.shape != moving_image.shape:
        raise ValueError("images must be same shape")

    src_freq = np.fft.fftn(reference_image)
    target_freq = np.fft.fftn(moving_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    eps = np.finfo(image_product.real.dtype).eps
    image_product /= (np.abs(image_product) + eps)
    cross_correlation = np.fft.ifftn(image_product)

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.stack(maxima).astype(np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        if return_error:
            src_amp = np.sum(np.real(src_freq * src_freq.conj()))
            src_amp /= src_freq.size
            target_amp = np.sum(np.real(target_freq * target_freq.conj()))
            target_amp /= target_freq.size
            CCmax = cross_correlation[maxima]
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        raise ValueError('Error: upsampled phase cross corrrelation not implemented here, use skimage.')

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    if return_error:
        error = np.real(1.0 - CCmax * CCmax.conj())
        phase_diff = np.arctan2(CCmax.imag, CCmax.real)
        return shifts, np.sqrt(np.abs(error)), phase_diff
    else:
        return shifts


def phase_cross_correlation_cv(reference_image, moving_image):
        """
        Compute openCV to compute the phase cross correlation. It is around 16 times faster than the implementation using
        numpy FFT (same as skimage).

        Parameters
        ----------
        reference_image : array
            Reference image.
        moving_image : array
            Image to register. Must be same dimensionality as
            ``reference_image``.

        Returns
        -------
        shifts : ndarray
            Shift vector (in pixels) required to register ``moving_image``
            with ``reference_image``. Axis ordering is consistent with
            numpy (e.g. Z, Y, X)
        error : float
            Peak response (see opencv description here:
            https://docs.opencv.org/4.5.3/d7/df3/group__imgproc__motion.html#ga552420a2ace9ef3fb053cd630fdb4952)
        """

        d, e = cv.phaseCorrelate(reference_image.astype(np.float32), moving_image.astype(np.float32))

        d_correct = [-np.round(d[1]).astype(int), -np.round(d[0]).astype(int)]

        return d_correct


def _compute_shift(reference_image, moving_image):
    """
    Backbone function to compute the registration and the registration error used for the global optimisation.
    This function can be replaced by experienced user to use their own registration and error estimation functions.

    Parameters
    ----------
    reference_image : array
        Reference image.
    moving_image : array
        Image to register. Must be same dimensionality as
        ``reference_image``.

    Returns
    -------
    d: array_like
        registration parameters found
    e: float
        error estimation for the registration (the higher the error the higher the registration uncertainty)
    """

    d = phase_cross_correlation_cv(reference_image, moving_image)
    e = max_sum_over_single_max(reference_image, moving_image, d)
    e = e/np.sqrt(np.mean(reference_image))*10

    return d, e


def _get_max_proj_apr(apr, parts, patch, patch_yx=None, plot=False):
    """
    Compute maximum projection on 3D APR data.

    Parameters
    ----------
    apr: pyapr.APR
        apr tree
    parts: pyapr.ParticlData
        apr particle
    patch: pyapr.ReconPatch
        patch for computing the projection only on the overlapping area.
    plot: bool
        control data plotting

    Returns
    -------
    _: list[ndarray]
        maximum intensity projection in each 3 dimension.
    """
    proj = []
    if patch_yx is None:
        for d in range(3):
            # dim=0: project along Y to produce a ZY plane
            # dim=1: project along X to produce a ZX plane
            # dim=2: project along Z to produce an YX plane
            proj.append(pyapr.transform.maximum_projection(apr, parts, dim=d, patch=patch, method='auto'))
    else:
        proj.append(pyapr.transform.projection.maximum_projection(apr, parts,
                                                                  dim=0, patch=patch, method='auto'))
        proj.append(pyapr.transform.projection.maximum_projection(apr, parts,
                                                                  dim=1, patch=patch, method='auto'))
        proj.append(pyapr.transform.projection.maximum_projection(apr, parts,
                                                                  dim=2, patch=patch_yx, method='auto'))

    if plot:
        fig, ax = plt.subplots(1, 3)
        for i, title in enumerate(['ZY', 'ZX', 'YX']):
            ax[i].imshow(proj[i], cmap='gray')
            ax[i].set_title(title)

    return proj[0], proj[1], proj[2]


def _get_proj_shifts(proj1, proj2):
    """
    This function computes shifts from max-projections on overlapping areas. It uses the phase cross-correlation
    to compute the shifts.

    Parameters
    ----------
    proj1: list[ndarray]
        max-projections for tile 1
    proj2: list[ndarray]
        max-projections for tile 2
    upsample_factor: float
        upsampling_factor for estimating the maximum phase cross-correlation position

    Returns
    -------
    _: array_like
        shifts in (x, y, z) and error measure (0=reliable, 1=not reliable)
    """
    # Compute phase cross-correlation to extract shifts
    dzy, error_zy = _compute_shift(proj1[0], proj2[0])
    dzx, error_zx = _compute_shift(proj1[1], proj2[1])
    dyx, error_yx = _compute_shift(proj1[2], proj2[2])

    # Replace error == 0 with 1 otherwise the minimum spanning tree considers that vertex are not connected
    if error_zy == 0:
        error_zy = 1e-6
    if error_zx == 0:
        error_zx = 1e-6
    if error_yx == 0:
        error_yx = 1e-6

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

    # for i, title, vector, err in zip(range(3), ['ZY', 'ZX', 'YX'], [dzy, dzx, dyx], [error_zy, error_zx, error_yx]):
    #     fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    #     ax[0].imshow(np.log(proj1[i]+1), cmap='gray')
    #     ax[0].set_title('d={}, e={:0.3f}'.format(vector, err))
    #     ax[1].imshow(np.log(proj2[i]+1), cmap='gray')
    #     ax[1].set_title(title)
    #
    #     shifted = warp(proj1[i], AffineTransform(translation=[vector[1], vector[0]]), mode='wrap', preserve_range=True)
    #     rgb = np.dstack((np.log(proj2[i]+1), np.log(shifted+1), np.zeros_like(proj1[i])))
    #     ax[2].imshow((rescale_intensity(rgb, out_range='uint8')).astype('uint8'))
    #
    # print('ok')

    return np.array([dz, dy, dx]), np.array([rz, ry, rx])


def _get_masked_proj_shifts(proj1, proj2, threshold, upsample_factor=1):
    """
    This function computes shifts from max-projections on overlapping areas with mask on brightest area.
    It uses the phase cross-correlation to compute the shifts.

    Parameters
    ----------
    proj1: list[ndarray]
        max-projections for tile 1
    proj2: list[ndarray]
        max-projections for tile 2
    upsample_factor: float
        upsampling_factor for estimating the maximum phase cross-correlation position

    Returns
    -------
    _: array_like
        shifts in (x, y, z) and error measure (0=reliable, 1=not reliable)
    """
    # Compute mask to discard very bright area that are likely bubbles or artefacts
    mask_ref = []
    mask_move = []
    for i in range(3):
        vmax = np.percentile(proj1[i], threshold)
        mask_ref.append(proj1[i] < vmax)
        vmax = np.percentile(proj2[i], threshold)
        mask_move.append(proj2[i] < vmax)

    # Compute phase cross-correlation to extract shifts
    dzy = phase_cross_correlation(proj1[0], proj2[0],
                                  return_error=True, upsample_factor=upsample_factor,
                                  reference_mask=mask_ref[0], moving_mask=mask_move[0])
    error_zy = np.sqrt(1 - correlate(proj1[0], proj2[0]).max() ** 2 / (np.sum(proj1 ** 2) * np.sum(proj2 ** 2)))
    dzx = phase_cross_correlation(proj1[1], proj2[1],
                                  return_error=True, upsample_factor=upsample_factor,
                                  reference_mask=mask_ref[1], moving_mask=mask_move[1])
    error_zx = np.sqrt(1 - correlate(proj1[1], proj2[1]).max() ** 2 / (np.sum(proj1 ** 2) * np.sum(proj2 ** 2)))
    dyx = phase_cross_correlation(proj1[2], proj2[2],
                                  return_error=True, upsample_factor=upsample_factor,
                                  reference_mask=mask_ref[2], moving_mask=mask_move[2])
    error_yx = np.sqrt(1 - correlate(proj1[2], proj2[2]).max() ** 2 / (np.sum(proj1 ** 2) * np.sum(proj2 ** 2)))

    # Replace error == 0 with 1e-6 otherwise the minimum spanning tree considers that vertex are not connected
    if error_zy == 0:
        error_zy = 1e-6
    if error_zx == 0:
        error_zx = 1e-6
    if error_yx == 0:
        error_yx = 1e-6

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

    # for i, title, vector in zip(range(3), ['ZY', 'ZX', 'YX'], [[dy, dz], [dx, dz], [dx, dy]]):
    #     fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    #     ax[0].imshow(proj1[i], cmap='gray')
    #     ax[0].set_title('dx={}, dy={}, dz={}'.format(dx, dy, dz))
    #     ax[1].imshow(proj2[i], cmap='gray')
    #     ax[1].set_title(title)
    #     from skimage.transform import warp, AffineTransform
    #     from skimage.exposure import rescale_intensity
    #     shifted = warp(proj1[i], AffineTransform(translation=vector), mode='wrap', preserve_range=True)
    #     rgb = np.dstack([proj2[i], shifted, np.zeros_like(proj1[i])])
    #     ax[2].imshow((rescale_intensity(rgb, out_range='uint8')).astype('uint8'))
    # print('ok')

    return np.array([dz, dy, dx]), np.array([rz, ry, rx])


class baseStitcher():
    """
    Base class for stitching multi-tile data.

    """
    def __init__(self,
                 tiles,
                 overlap_h: (int, float),
                 overlap_v: (int, float)):
        """
        Constructor for the baseStitcher class.

        Parameters
        ----------
        tiles: tileParser
            tileParser object containing the dataset to stitch.
        overlap_h: float
            expected horizontal overlap in %
        overlap_v: float
            expected vertical overlap in %

        Returns
        -------
        None
        """
        self.tiles = tiles
        self.ncol = tiles.ncol
        self.nrow = tiles.nrow
        self.n_vertex = tiles.n_tiles
        self.n_edges = tiles.n_edges
        self.frame_size = tiles.frame_size

        self.expected_overlap_h = int(overlap_h/100*self.frame_size)
        self.expected_overlap_v = int(overlap_v/100*self.frame_size)

        self.overlap_h = int(self.expected_overlap_h*1.2)
        if self.expected_overlap_h > self.frame_size:
            self.expected_overlap_h = self.frame_size
        self.overlap_v = int(self.expected_overlap_v*1.2)
        if self.expected_overlap_v > self.frame_size:
            self.expected_overlap_v = self.frame_size

        self.mask = False
        self.threshold = None

        self.segment = False
        self.segmenter = None

        self.reg_x = int(self.frame_size*0.05)
        self.reg_y = int(self.frame_size*0.05)
        self.reg_z = 20

        self.z_begin = None
        self.z_end = None

    def activate_mask(self, threshold):
        """
        Activate the masked cross-correlation for the displacement estimation. Pixels above threshold are
        not taken into account.

        Parameters
        ----------
        threshold: int
            threshold for the cross-correlation mask as a percentage of pixel to keep (e.g. 95 will create a mask
            removing the 5% brightest pixels).

        """
        self.mask = True
        self.threshold = threshold

    def deactivate_mask(self):
        """
        Deactivate the masked cross-correlation and uses a classical cross correlation.

        """
        self.mask = False
        self.threshold = None

    def save_database(self, path=None):
        """
        Save database at the given path. The database must be built before calling this method.

        Parameters
        ----------
        path: string
            path to save the database.

        """

        if self.database is None:
            raise TypeError('Error: database can''t be saved because it was not created. '
                            'Please call build_database() first.')

        if path is None:
            path = os.path.join(self.tiles.path, 'registration_results.csv')

        self.database.to_csv(path)

    def load_database(self, path=None, force=False):
        """
        Save database at the given path. The database must be built before calling this method.

        Parameters
        ----------
        path: string
            path to save the database.

        """

        if self.database is not None and force is False:
            raise TypeError('Error: database can''t be read because a database is already exist and force is set'
                            'to False.')

        if path is None:
            path = os.path.join(self.tiles.path, 'registration_results.csv')

        self.database = pd.read_csv(path)

    def activate_segmentation(self, segmenter):
        """
        Activate the segmentation. When a tile is loaded it is segmented before the stitching is done.

        Parameters
        ----------
        segmenter: tileSegmenter
            segmenter object for segmenting each tile.

        """
        self.segment = True
        self.segmenter = segmenter

    def deactivate_segmentation(self):
        """
        Deactivate tile segmentation.

        """

        self.segment = False

    def reconstruct_slice(self, loc=None, n_proj=0, dim=0, downsample=1, color=False, debug=False, plot=True, seg=False, progress_bar=True):
        """
        Reconstruct whole sample 2D section at the given location and in a given dimension. This function can also
        reconstruct a maximum intensity projection if `n_proj>0`.

        Parameters
        ----------
        loc: int (default: middle of the sample)
            Position of the plane where the reconstruction should be done. The location varies depending on the
            downsample parameter and should be adapted.
        n_proj: int (default: 0)
            Number of planes to perform the maximum intensity projection.
        dim: int (default: 0)
            Dimension of the reconstruction, e.g. 0 will be [y, x] plane (orthogonal to z).
        downsample: int (default: 1)
            Downsample factor for the reconstruction. Must be in [1, 2, 4, 8, 16, 32].
        color: bool (default: False)
            Option to reconstruct with checkerboard color pattern. Useful to identify doubling artifacts.
        debug: bool (default: False)
            Option to add a white square for each tile, making it easy to see overlapping areas.
        plot: bool (default: True)
            Define if the function plots the results with Matplotlib or just returns an array.
        seg: bool (default: False)
            Option to also reconstruct the segmentation. Only works with `dim=0`

        Returns
        -------
        _: ndarray
            Array containing the reconstructed data.
        """

        if dim == 0:
            return self._reconstruct_z_slice(z=loc, n_proj=n_proj, downsample=downsample, color=color,
                                             debug=debug, plot=plot, seg=seg, progress_bar=progress_bar)
        elif dim == 1:
            return self._reconstruct_y_slice(y=loc, n_proj=n_proj, downsample=downsample, color=color,
                                             debug=debug, plot=plot, progress_bar=progress_bar)
        elif dim == 2:
            return self._reconstruct_x_slice(x=loc, n_proj=n_proj, downsample=downsample, color=color,
                                             debug=debug, plot=plot, progress_bar=progress_bar)
        else:
            raise ValueError('dim should be in [1, 2, 3], got dim = {}'.format(dim))

    def set_regularization(self, reg_x, reg_y, reg_z):
        """
        Set the regularization for the stitching to prevent aberrant displacements.

        Parameters
        ----------
        reg_x: int
            if the horizontal displacement computed in the pairwise registration for any tile is greater than
            reg_x (in pixel unit) then the expected displacement (from motor position) is taken.
        reg_y: int
            if the horizontal displacement computed in the pairwise registration for any tile is greater than
            reg_z (in pixel unit) then the expected displacement (from motor position) is taken.
        reg_z: int
            if the horizontal displacement computed in the pairwise registration for any tile is greater than
            reg_z (in pixel unit) then the expected displacement (from motor position) is taken.

        Returns
        -------
        None
        """

        self.reg_x = reg_x
        self.reg_y = reg_y
        self.reg_z = reg_z

    def reconstruct_z_color(self, z=None, n_proj=10, downsample=1, debug=False, plot=True, progress_bar=True):
        """
        Reconstruct and merge the sample at a given depth z.

        Parameters
        ----------
        z: int
            reconstruction depth
        downsample: int
            downsample for reconstruction (must be a power of 2)
        debug: bool
            if true the border of each tile will be highlighted

        Returns
        -------
        merged_data: ndarray
            Merged frame at depth z.
        """

        level_delta = int(-np.sign(downsample) * np.log2(np.abs(downsample)))

        tile = self.tiles[0]
        tile.lazy_load_tile(level_delta=level_delta)

        if z is None:
            z = int(tile.lazy_data.shape[0] / 2)

        if z > tile.lazy_data.shape[0]:
            raise ValueError('Error: z is too large ({}), maximum depth at this downsample is {}.'.format(z, tile.lazy_data.shape[0]))

        frame_size = tile.lazy_data.shape[1:]
        x_pos = self.database['ABS_H'].to_numpy()
        nx = int(np.ceil((x_pos.max() - x_pos.min()) / downsample + frame_size[1]))
        y_pos = self.database['ABS_V'].to_numpy()
        ny = int(np.ceil((y_pos.max() - y_pos.min()) / downsample + frame_size[0]))

        H = np.zeros((ny, nx), dtype='uint16')
        V = np.zeros((ny, nx), dtype='uint16')

        H_pos = (x_pos - x_pos.min()) / downsample
        V_pos = (y_pos - y_pos.min()) / downsample

        for i, tile in enumerate(tqdm(self.tiles, desc='Merging', disable=not progress_bar)):
            tile.lazy_load_tile(level_delta=level_delta)
            zf = min(z+n_proj, tile.lazy_data.shape[0])
            data = tile.lazy_data[z:zf]
            v = data.max(axis=0)
            h = np.argmax(data, axis=0)

            # In debug mode we highlight each tile edge to see where it was
            if debug:
                v[self.overlap_v, :] = 2**16-1
                v[-self.overlap_v, :] = 2**16-1
                v[:, self.overlap_h] = 2**16-1
                v[:, -self.overlap_h] = 2**16-1

            x1 = int(H_pos[i])
            x2 = int(H_pos[i] + v.shape[1])
            y1 = int(V_pos[i])
            y2 = int(V_pos[i] + v.shape[0])

            V[y1:y2, x1:x2] = np.maximum(V[y1:y2, x1:x2], v)
            H[y1:y2, x1:x2] = np.maximum(H[y1:y2, x1:x2], h)

        H = rescale_intensity(gaussian(H, sigma=2/downsample), out_range=np.float64)*0.66
        V = np.log(V + 200)
        vmin, vmax = np.percentile(V[V > np.log(100)], (1, 99.9))
        V = rescale_intensity(V, in_range=(vmin, vmax), out_range=np.float64)
        S = rescale_intensity(V**1.5, out_range=np.float64)*0.66
        rgb = hsv2rgb(np.dstack((H, S, V)))
        rgb = rescale_intensity(rgb, out_range='uint8')

        if plot:
            fig, ax = plt.subplots(1, 1)
            h = ax.imshow(rgb, cmap='turbo', vmin=0, vmax=n_proj*downsample)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, orientation='vertical', label='Depth [pixel]')

        return rgb

    def _reconstruct_z_slice(self, z=None, n_proj=0, downsample=1, color=False,
                             debug=False, plot=True, seg=False, progress_bar=True):
        """
        Reconstruct and merge the sample at a given depth z.

        Parameters
        ----------
        z: int
            reconstruction depth (vary with downsample)
        n_proj: int (default: 0)
            Number of planes to perform the maximum intensity projection.
        dim: int (default: 0)
            Dimension of the reconstruction, e.g. 0 will be [y, x] plane (orthogonal to z).
        downsample: int (default: 1)
            Downsample factor for the reconstruction. Must be in [1, 2, 4, 8, 16, 32].
        color: bool (default: False)
            Option to reconstruct with checkerboard color pattern. Useful to identify doubling artifacts.
        debug: bool (default: False)
            Option to add a white square for each tile, making it easy to see overlapping areas.
        plot: bool (default: True)
            Define if the function plots the results with Matplotlib or just returns an array.
        seg: bool (default: False)
            Option to also reconstruct the segmentation. Only works with `dim=0`

        Returns
        -------
        merged_data: ndarray
            Merged frame at depth z.
        """

        level_delta = int(-np.sign(downsample) * np.log2(np.abs(downsample)))

        tile = self.tiles[0]
        tile.lazy_load_tile(level_delta=level_delta)
        if seg:
            tile.lazy_load_segmentation(level_delta=level_delta)

        if z is None:
            z = int(tile.lazy_data.shape[0] / 2)

        if z > tile.lazy_data.shape[0]:
            raise ValueError('Error: z is too large ({}), maximum depth at this downsample is {}.'.format(z, tile.lazy_data.shape[0]))

        frame_size = tile.lazy_data.shape[1:]
        x_pos = self.database['ABS_H'].to_numpy()
        nx = int(np.ceil((x_pos.max() - x_pos.min()) / downsample + frame_size[1]))
        y_pos = self.database['ABS_V'].to_numpy()
        ny = int(np.ceil((y_pos.max() - y_pos.min()) / downsample + frame_size[0]))

        if color:
            merged_data = np.ones((ny, nx, 3), dtype='uint16')
            merged_data[:, :, 2] = 0
        else:
            merged_data = np.zeros((ny, nx), dtype='uint16')
            if seg:
                merged_seg = np.zeros((ny, nx), dtype='float32')

        H_pos = (x_pos - x_pos.min()) / downsample
        V_pos = (y_pos - y_pos.min()) / downsample

        for i, tile in enumerate(tqdm(self.tiles, desc='Merging', disable=not progress_bar)):
            tile.lazy_load_tile(level_delta=level_delta)
            if seg:
                tile.lazy_load_segmentation(level_delta=level_delta)
            zf = min(z+n_proj, tile.lazy_data.shape[0])
            if zf > z:
                data = tile.lazy_data[z:zf].max(axis=0)
                if seg:
                    cc = tile.lazy_segmentation[z:zf].max(axis=0)
            else:
                data = tile.lazy_data[z]
                if seg:
                    cc = tile.lazy_segmentation[z]

            # In debug mode we highlight each tile edge to see where it was
            if debug:
                xv = int(self.expected_overlap_v/downsample)
                xh = int(self.expected_overlap_h/downsample)
                data[xv, xh:-xh] = 2**16-1
                data[-xv, xh:-xh] = 2**16-1
                data[xv:-xv, xh] = 2**16-1
                data[xv:-xv, -xh] = 2**16-1

            x1 = int(H_pos[i])
            x2 = int(H_pos[i] + data.shape[1])
            y1 = int(V_pos[i])
            y2 = int(V_pos[i] + data.shape[0])

            if color:
                if tile.col % 2:
                    if tile.row % 2:
                        merged_data[y1:y2, x1:x2, 0] = np.maximum(merged_data[y1:y2, x1:x2, 1], data)
                    else:
                        merged_data[y1:y2, x1:x2, 1] = np.maximum(merged_data[y1:y2, x1:x2, 0], data)
                else:
                    if tile.row % 2:
                        merged_data[y1:y2, x1:x2, 1] = np.maximum(merged_data[y1:y2, x1:x2, 1], data)
                    else:
                        merged_data[y1:y2, x1:x2, 0] = np.maximum(merged_data[y1:y2, x1:x2, 0], data)
            else:
                merged_data[y1:y2, x1:x2] = np.maximum(merged_data[y1:y2, x1:x2], data)
                if seg:
                    merged_seg[y1:y2, x1:x2] = np.maximum(merged_seg[y1:y2, x1:x2], cc)

        if plot:
            viewer = napari.Viewer()
            if color:
                viewer.add_image(self._process_RGB_for_display(merged_data), name='Z plane')
            else:
                viewer.add_image(self._process_GRAY_for_display(merged_data), name='Z plane', contrast_limits=[0, 2**16-1])
                if seg:
                    viewer.add_labels(merged_seg, name='Segmentation labels.')

        if not seg:
            return merged_data
        else:
            return merged_data, merged_seg

    def _reconstruct_y_slice(self, y=None, n_proj=0, downsample=1, color=False,
                             debug=False, plot=True, progress_bar=True):
        """
        Reconstruct and merge the sample at a given position y.

        Parameters
        ----------
        y: int
            reconstruction location in y
        n_proj: int (default: 0)
            Number of planes to perform the maximum intensity projection.
        dim: int (default: 0)
            Dimension of the reconstruction, e.g. 0 will be [y, x] plane (orthogonal to z).
        downsample: int (default: 1)
            Downsample factor for the reconstruction. Must be in [1, 2, 4, 8, 16, 32].
        color: bool (default: False)
            Option to reconstruct with checkerboard color pattern. Useful to identify doubling artifacts.
        debug: bool (default: False)
            Option to add a white square for each tile, making it easy to see overlapping areas.
        plot: bool (default: True)
            Define if the function plots the results with Matplotlib or just returns an array.

        Returns
        -------
        merged_data: ndarray
            Merged frame at position y.
        """

        level_delta = int(-np.sign(downsample) * np.log2(np.abs(downsample)))

        tile = self.tiles[0]
        tile.lazy_load_tile(level_delta=level_delta)
        tile_shape = tile.lazy_data.shape

        if y is None:
            y = int(tile_shape[1]*self.tiles.nrow/2)

        if y > tile.lazy_data.shape[1]*self.tiles.nrow:
            raise ValueError('Error: y is too large ({}), maximum depth at this downsample is {}.'
                             .format(y, tile.lazy_data.shape[1]*self.tiles.nrow))

        x_pos = self.database['ABS_H'].to_numpy()
        nx = int(np.ceil((x_pos.max() - x_pos.min()) / downsample + tile_shape[2]))
        y_pos = self.database['ABS_V'].to_numpy()
        ny = int(np.ceil((y_pos.max() - y_pos.min()) / downsample + tile_shape[1]))
        z_pos = self.database['ABS_D'].to_numpy()
        nz = int(np.ceil((z_pos.max() - z_pos.min()) / downsample + tile_shape[0]))

        # Determine tiles to load
        tiles_to_load = []
        tiles_pos = []
        for x_loc, y_loc, z_loc, tile in zip(x_pos/downsample, y_pos/downsample, z_pos/downsample, self.tiles):
            if (y > y_loc) and (y < y_loc+tile_shape[1]):
                tiles_to_load.append(tile)
                tiles_pos.append([z_loc, y_loc, x_loc])
        tiles_pos = np.array(tiles_pos).astype('uint64')

        if color:
            merged_data = np.ones((nz, nx, 3), dtype='uint16')
            merged_data[:, :, 2] = 0
        else:
            merged_data = np.zeros((nz, nx), dtype='uint16')

        for i, tile in enumerate(tqdm(tiles_to_load, desc='Merging', disable=not progress_bar)):
            tile.lazy_load_tile(level_delta=level_delta)
            y_tile = int(y - tiles_pos[i, 1])
            yf = min(y_tile+n_proj, tiles_pos[i, 1]+tile.lazy_data.shape[1])
            if yf > y:
                data = tile.lazy_data[:, y_tile:yf, :].max(axis=1)
            else:
                data = tile.lazy_data[:, y_tile, :]

            # In debug mode we highlight each tile edge to see where it was
            if debug:
                xv = int(self.expected_overlap_v/downsample)
                xh = int(self.expected_overlap_h/downsample)
                data[xv, xh:-xh] = 2**16-1
                data[-xv, xh:-xh] = 2**16-1
                data[xv:-xv, xh] = 2**16-1
                data[xv:-xv, -xh] = 2**16-1


            x1 = int(tiles_pos[i, 2])
            x2 = int(tiles_pos[i, 2] + data.shape[1])
            z1 = int(tiles_pos[i, 0])
            z2 = int(tiles_pos[i, 0] + data.shape[0])
            
            if color:
                if tile.col % 2:
                    if tile.row % 2:
                        merged_data[z1:z2, x1:x2, 0] = np.maximum(merged_data[z1:z2, x1:x2, 1], data)
                    else:
                        merged_data[z1:z2, x1:x2, 1] = np.maximum(merged_data[z1:z2, x1:x2, 0], data)
                else:
                    if tile.row % 2:
                        merged_data[z1:z2, x1:x2, 1] = np.maximum(merged_data[z1:z2, x1:x2, 1], data)
                    else:
                        merged_data[z1:z2, x1:x2, 0] = np.maximum(merged_data[z1:z2, x1:x2, 0], data)
            else:
                merged_data[z1:z2, x1:x2] = np.maximum(merged_data[z1:z2, x1:x2], data)

        if plot:
            viewer = napari.Viewer()
            if color:
                viewer.add_image(self._process_RGB_for_display(merged_data), name='Y plane')
            else:
                viewer.add_image(self._process_GRAY_for_display(merged_data), name='Y plane')

        return merged_data

    def _reconstruct_x_slice(self, x=None, n_proj=0, downsample=1, color=False,
                             debug=False, plot=True, progress_bar=True):
        """
        Reconstruct and merge the sample at a given position x.

        Parameters
        ----------
        x: int
            reconstruction location in x
        n_proj: int (default: 0)
            Number of planes to perform the maximum intensity projection.
        dim: int (default: 0)
            Dimension of the reconstruction, e.g. 0 will be [y, x] plane (orthogonal to z).
        downsample: int (default: 1)
            Downsample factor for the reconstruction. Must be in [1, 2, 4, 8, 16, 32].
        color: bool (default: False)
            Option to reconstruct with checkerboard color pattern. Useful to identify doubling artifacts.
        debug: bool (default: False)
            Option to add a white square for each tile, making it easy to see overlapping areas.
        plot: bool (default: True)
            Define if the function plots the results with Matplotlib or just returns an array.

        Returns
        -------
        merged_data: ndarray
            Merged frame at position x.
        """

        level_delta = int(-np.sign(downsample) * np.log2(np.abs(downsample)))

        tile = self.tiles[0]
        tile.lazy_load_tile(level_delta=level_delta)
        tile_shape = tile.lazy_data.shape

        if x is None:
            x = int(tile_shape[2]*self.tiles.ncol/2)

        if x > tile.lazy_data.shape[2]*self.tiles.ncol:
            raise ValueError('Error: y is too large ({}), maximum depth at this downsample is {}.'
                             .format(x, tile.lazy_data.shape[2]*self.tiles.ncol))

        x_pos = self.database['ABS_H'].to_numpy()
        nx = int(np.ceil((x_pos.max() - x_pos.min()) / downsample + tile_shape[2]))
        y_pos = self.database['ABS_V'].to_numpy()
        ny = int(np.ceil((y_pos.max() - y_pos.min()) / downsample + tile_shape[1]))
        z_pos = self.database['ABS_D'].to_numpy()
        nz = int(np.ceil((z_pos.max() - z_pos.min()) / downsample + tile_shape[0]))

        # Determine tiles to load
        tiles_to_load = []
        tiles_pos = []
        for x_loc, y_loc, z_loc, tile in zip(x_pos/downsample, y_pos/downsample, z_pos/downsample, self.tiles):
            if (x > x_loc) and (x < x_loc+tile_shape[2]):
                tiles_to_load.append(tile)
                tiles_pos.append([z_loc, y_loc, x_loc])
        tiles_pos = np.array(tiles_pos).astype('uint64')

        if color:
            merged_data = np.ones((nz, ny, 3), dtype='uint16')
            merged_data[:, :, 2] = 0
        else:
            merged_data = np.zeros((nz, ny), dtype='uint16')

        for i, tile in enumerate(tqdm(tiles_to_load, desc='Merging', disable=not progress_bar)):
            tile.lazy_load_tile(level_delta=level_delta)
            x_tile = int(x - tiles_pos[i, 2])
            xf = min(x_tile+n_proj, tiles_pos[i, 2]+tile.lazy_data.shape[2])
            if xf > x:
                data = tile.lazy_data[:, :, x_tile:xf].max(axis=2)
            else:
                data = tile.lazy_data[:, :, x_tile]

            # In debug mode we highlight each tile edge to see where it was
            if debug:
                xv = int(self.expected_overlap_v/downsample)
                xh = int(self.expected_overlap_h/downsample)
                data[xv, xh:-xh] = 2**16-1
                data[-xv, xh:-xh] = 2**16-1
                data[xv:-xv, xh] = 2**16-1
                data[xv:-xv, -xh] = 2**16-1


            y1 = int(tiles_pos[i, 1])
            y2 = int(tiles_pos[i, 1] + data.shape[1])
            z1 = int(tiles_pos[i, 0])
            z2 = int(tiles_pos[i, 0] + data.shape[0])
            
            if color:
                if tile.col % 2:
                    if tile.row % 2:
                        merged_data[z1:z2, y1:y2, 0] = np.maximum(merged_data[z1:z2, y1:y2, 1], data)
                    else:
                        merged_data[z1:z2, y1:y2, 1] = np.maximum(merged_data[z1:z2, y1:y2, 0], data)
                else:
                    if tile.row % 2:
                        merged_data[z1:z2, y1:y2, 1] = np.maximum(merged_data[z1:z2, y1:y2, 1], data)
                    else:
                        merged_data[z1:z2, y1:y2, 0] = np.maximum(merged_data[z1:z2, y1:y2, 0], data)
            else:
                merged_data[z1:z2, y1:y2] = np.maximum(merged_data[z1:z2, y1:y2], data)

        if plot:
            viewer = napari.Viewer()
            if color:
                viewer.add_image(self._process_RGB_for_display(merged_data), name='X plane')
            else:
                viewer.add_image(self._process_GRAY_for_display(merged_data), name='X plane')

        return merged_data

    def _process_RGB_for_display(self, u):
        """
        Process RGB data for correctly displaying it.

        Parameters
        ----------
        u: ndarray
            RGB data

        Returns
        -------
        data_to_display: ndarray
            RGB data displayable with correct contrast and colors.
        """
        data_to_display = np.zeros_like(u, dtype='uint8')
        for i in range(2):
            tmp = np.log(u[:, :, i] + 200)
            vmin, vmax = np.percentile(tmp[tmp > np.log(1 + 200)], (1, 99.9))
            data_to_display[:, :, i] = rescale_intensity(tmp, in_range=(vmin, vmax), out_range='uint8')

        return data_to_display

    def _process_GRAY_for_display(self, u):
        """
        Process RGB data for correctly displaying it.

        Parameters
        ----------
        u: ndarray
            RGB data

        Returns
        -------
        data_to_display: ndarray
            RGB data displayable with correct contrast and colors.
        """
        u = np.log(u+200)
        vmin, vmax = np.percentile(u[u > np.log(1 + 200)], (1, 99.9))
        data_to_display = rescale_intensity(u, in_range=(vmin, vmax), out_range='uint16')

        return data_to_display

    def _regularize(self, reg, rel):
        """
        Remove too large displacement and replace them with expected one with a large uncertainty.

        """
        if np.abs(reg[2] - (self.overlap_h - self.expected_overlap_h)) > self.reg_x:
            reg[2] = (self.overlap_h - self.expected_overlap_h)
            rel[2] = 2
        if np.abs(reg[1] - (self.overlap_v - self.expected_overlap_v)) > self.reg_y:
            reg[1] = (self.overlap_v - self.expected_overlap_v)
            rel[1] = 2
        if np.abs(reg[0]) > self.reg_z:
            reg[0] = 0
            rel[0] = 2

        return reg, rel

    def _save_max_projs(self):
        """
        Save the computed maximum intensity projection on persistent memory. This is useful to recompute the
        registration directly from the max. proj. but only works if the overlaps are kept the same.

        Returns
        -------
        None
        """
        # Safely create folder to save max-projs
        Path(self.tiles.folder_max_projs).mkdir(parents=True, exist_ok=True)

        for row in range(self.nrow):
            for col in range(self.ncol):
                proj = self.projs[row, col]
                if proj is not None:
                    for loc in proj.keys():
                        for i, d in enumerate(['zy', 'zx', 'yx']):
                            data = proj[loc]
                            np.save(os.path.join(self.tiles.folder_max_projs,
                                                 '{}_{}_{}_{}.npy'.format(row, col, loc, d)), data[i])

    def _load_max_projs(self, path):
        """
        Load the maximum intensity projection previously stored.

        Parameters
        ----------
        path: str
            path to load the maximum intensity projection from. If None then default to `max_projs` folder in the
            acquisition folder.

        Returns
        -------
        None
        """
        projs = np.empty((self.nrow, self.ncol), dtype=object)
        if path is None:
            folder_max_projs = self.tiles.folder_max_projs
        else:
            folder_max_projs = path

        for tile in self.tiles:
            proj = {}
            if tile.col + 1 < self.tiles.ncol:
                if self.tiles.tiles_pattern[tile.row, tile.col + 1] == 1:
                    # EAST 1
                    tmp = []
                    for i, d in enumerate(['zy', 'zx', 'yx']):
                        tmp.append(np.load(os.path.join(folder_max_projs,
                                                        '{}_{}_east_{}.npy'.format(tile.row, tile.col, d))))
                    proj['east'] = tmp
            if tile.col - 1 >= 0:
                if self.tiles.tiles_pattern[tile.row, tile.col - 1] == 1:
                    # EAST 2
                    tmp = []
                    for i, d in enumerate(['zy', 'zx', 'yx']):
                        tmp.append(np.load(os.path.join(folder_max_projs,
                                                        '{}_{}_west_{}.npy'.format(tile.row, tile.col, d))))
                    proj['west'] = tmp
            if tile.row + 1 < self.tiles.nrow:
                if self.tiles.tiles_pattern[tile.row + 1, tile.col] == 1:
                    # SOUTH 1
                    tmp = []
                    for i, d in enumerate(['zy', 'zx', 'yx']):
                        tmp.append(np.load(os.path.join(folder_max_projs,
                                                        '{}_{}_south_{}.npy'.format(tile.row, tile.col, d))))
                    proj['south'] = tmp
            if tile.row - 1 >= 0:
                if self.tiles.tiles_pattern[tile.row - 1, tile.col] == 1:
                    # SOUTH 2
                    tmp = []
                    for i, d in enumerate(['zy', 'zx', 'yx']):
                        tmp.append(np.load(os.path.join(folder_max_projs,
                                                        '{}_{}_north_{}.npy'.format(tile.row, tile.col, d))))
                    proj['north'] = tmp

            projs[tile.row, tile.col] = proj

        return projs

    def _precompute_max_projs(self, progress_bar=True):
        """
        Precompute max-projections for loading the data only once during the stitching.

        Returns
        -------
        None
        """
        projs = np.empty((self.nrow, self.ncol), dtype=object)
        for tile in tqdm(self.tiles, desc='Computing max. proj.', disable=not progress_bar):
            tile.load_tile()
            proj = {}
            if tile.col + 1 < self.tiles.ncol:
                if self.tiles.tiles_pattern[tile.row, tile.col + 1] == 1:
                    # EAST 1
                    patch = pyapr.ReconPatch()
                    patch.y_begin = self.frame_size - self.overlap_h
                    if self.z_begin is None:
                        proj['east'] = _get_max_proj_apr(tile.apr, tile.parts, patch, plot=False)
                    else:
                        patch_yx = pyapr.ReconPatch()
                        patch_yx.y_begin = self.frame_size - self.overlap_h
                        patch_yx.z_begin = self.z_begin
                        patch_yx.z_end = self.z_end
                        proj['east'] = _get_max_proj_apr(tile.apr, tile.parts, patch=patch, patch_yx=patch_yx,
                                                              plot=False)
            if tile.col - 1 >= 0:
                if self.tiles.tiles_pattern[tile.row, tile.col - 1] == 1:
                    # EAST 2
                    patch = pyapr.ReconPatch()
                    patch.y_end = self.overlap_h
                    if self.z_begin is None:
                        proj['west'] = _get_max_proj_apr(tile.apr, tile.parts, patch, plot=False)
                    else:
                        patch_yx = pyapr.ReconPatch()
                        patch_yx.y_end = self.overlap_h
                        patch_yx.z_begin = self.z_begin
                        patch_yx.z_end = self.z_end
                        proj['west'] = _get_max_proj_apr(tile.apr, tile.parts, patch=patch, patch_yx=patch_yx,
                                                              plot=False)
            if tile.row + 1 < self.tiles.nrow:
                if self.tiles.tiles_pattern[tile.row + 1, tile.col] == 1:
                    # SOUTH 1
                    patch = pyapr.ReconPatch()
                    patch.x_begin = self.frame_size - self.overlap_v
                    if self.z_begin is None:
                        proj['south'] = _get_max_proj_apr(tile.apr, tile.parts, patch, plot=False)
                    else:
                        patch_yx = pyapr.ReconPatch()
                        patch_yx.x_begin = self.frame_size - self.overlap_v
                        patch_yx.z_begin = self.z_begin
                        patch_yx.z_end = self.z_end
                        proj['south'] = _get_max_proj_apr(tile.apr, tile.parts, patch=patch, patch_yx=patch_yx,
                                                               plot=False)
            if tile.row - 1 >= 0:
                if self.tiles.tiles_pattern[tile.row - 1, tile.col] == 1:
                    # SOUTH 2
                    patch = pyapr.ReconPatch()
                    patch.x_end = self.overlap_v
                    if self.z_begin is None:
                        proj['north'] = _get_max_proj_apr(tile.apr, tile.parts, patch, plot=False)
                    else:
                        patch_yx = pyapr.ReconPatch()
                        patch_yx.x_end = self.overlap_v
                        patch_yx.z_begin = self.z_begin
                        patch_yx.z_end = self.z_end
                        proj['north'] = _get_max_proj_apr(tile.apr, tile.parts, patch=patch, patch_yx=patch_yx,
                                                               plot=False)

            projs[tile.row, tile.col] = proj

            if self.segment:
                self.segmenter.compute_segmentation(tile)

        self.projs = projs


class tileStitcher(baseStitcher):
    """
    Class used to perform the stitching. The stitching is performed in 4 steps:

    1. The pairwise registration parameters of each neighboring tile is computed on the max-projection
    2. A sparse graph (edges = tiles and vertex = registration between neighboring tiles) is constructed to store
       the registration parameters (displacements and reliability)
    3. The sparse graph is optimized to satisfy the constraints (every loop in the graph should sum to 0) using the
       maximum spanning tree on the reliability estimation.
    4. The maximum spanning tree is parsed to extract optimal tile positions solution.

    The beauty of this method is that it scales well with increasing dataset sizes and because the final optimization
    is very fast and does not require to reload the data.

    """
    def __init__(self,
                 tiles,
                 overlap_h: (int, float),
                 overlap_v: (int, float)):
        """
        Constructor for the tileStitcher class.

        Parameters
        ----------
        tiles: tileParser
            tileParser object containing the dataset to stitch.
        overlap_h: float
            expected horizontal overlap in %
        overlap_v: float
            expected vertical overlap in %
        """

        super().__init__(tiles, overlap_h, overlap_v)

        self.cgraph_from = []
        self.cgraph_to = []
        self.relia_H = []
        self.relia_V = []
        self.relia_D = []
        self.dH = []
        self.dV = []
        self.dD = []

        # Attributes below are set when the corresponding method are called.
        self.registration_map_rel = None
        self.registration_map_abs = None
        self.ctree_from_H = None
        self.ctree_from_V = None
        self.ctree_from_D = None
        self.ctree_to_H = None
        self.ctree_to_V = None
        self.ctree_to_D = None
        self.min_tree_H = None
        self.min_tree_V = None
        self.min_tree_D = None
        self.graph_relia_H = None
        self.graph_relia_V = None
        self.graph_relia_D = None
        self.database = None
        self.projs = None

    def _compute_registration_old(self):
        """
        Compute the pair-wise registration for all tiles. This implementation loads the data twice and is therefore
        not efficient.

        """
        for tile in tqdm(self.tiles, desc='Computing stitching'):
            tile.load_tile()
            tile.load_neighbors()

            if self.segment:
                self.segmenter.compute_segmentation(tile)

            for apr, parts, coords in zip(tile.apr_neighbors, tile.parts_neighbors, tile.neighbors):

                if tile.row == coords[0] and tile.col < coords[1]:
                    # EAST
                    reg, rel = self._compute_east_registration(tile.apr, tile.parts, apr, parts)

                elif tile.col == coords[1] and tile.row < coords[0]:
                    # SOUTH
                    reg, rel = self._compute_south_registration(tile.apr, tile.parts, apr, parts)

                else:
                    raise TypeError('Error: couldn''t determine registration to perform.')

                # Regularize in cas of aberrant displacements
                reg, rel = self._regularize(reg, rel)

                self.cgraph_from.append(np.ravel_multi_index([tile.row, tile.col],
                                                                    dims=(self.nrow, self.ncol)))
                self.cgraph_to.append(np.ravel_multi_index([coords[0], coords[1]],
                                                                  dims=(self.nrow, self.ncol)))
                # H=x, V=y, D=z
                self.dH.append(reg[2])
                self.dV.append(reg[1])
                self.dD.append(reg[0])
                self.relia_H.append(rel[2])
                self.relia_V.append(rel[1])
                self.relia_D.append(rel[0])

        self._build_sparse_graphs()
        self._optimize_sparse_graphs()
        _, _ = self._produce_registration_map()
        self._build_database()
        self._print_info()

    def compute_registration(self, on_disk=False, progress_bar=True):
        """
        Compute the pair-wise registration for all tiles. This implementation loads the data once by precomputing
        the max-proj and is therefore efficient.

        """
        # First we pre-compute the max-projections and keep them in memory or save them on disk and load them up.
        if on_disk:
            # It makes more sens and it avoids loading the max-proj when it is computed.
            self._precompute_max_projs(progress_bar=progress_bar)
            self._save_max_projs()
        else:
            self._precompute_max_projs(progress_bar=progress_bar)

        # Then we loop again through the tiles but now we have access to the max-proj
        for tile in tqdm(self.tiles, desc='Computing cross-correlations', disable=not progress_bar):
            proj1 = self.projs[tile.row, tile.col]

            for coords in tile.neighbors:
                proj2 = self.projs[coords[0], coords[1]]

                if tile.row == coords[0] and tile.col < coords[1]:
                    # EAST
                    if self.mask:
                        reg, rel = _get_masked_proj_shifts(proj1['east'], proj2['west'], threshold=self.threshold)
                    else:
                        reg, rel = _get_proj_shifts(proj1['east'], proj2['west'])

                elif tile.col == coords[1] and tile.row < coords[0]:
                    # SOUTH
                    if self.mask:
                        reg, rel = _get_masked_proj_shifts(proj1['south'], proj2['north'], threshold=self.threshold)
                    else:
                        reg, rel = _get_proj_shifts(proj1['south'], proj2['north'])

                else:
                    raise TypeError('Error: couldn''t determine registration to perform.')

                self.cgraph_from.append(np.ravel_multi_index([tile.row, tile.col],
                                                                    dims=(self.nrow, self.ncol)))
                self.cgraph_to.append(np.ravel_multi_index([coords[0], coords[1]],
                                                                  dims=(self.nrow, self.ncol)))

                # Regularize in case of aberrant displacements
                reg, rel = self._regularize(reg, rel)

                # H=x, V=y, D=z
                self.dH.append(reg[2])
                self.dV.append(reg[1])
                self.dD.append(reg[0])
                self.relia_H.append(rel[2])
                self.relia_V.append(rel[1])
                self.relia_D.append(rel[0])

        self._build_sparse_graphs()
        self._optimize_sparse_graphs()
        _, _ = self._produce_registration_map()
        self._build_database()
        self._print_info()

    def compute_registration_from_max_projs(self, path=None):
        """
        Compute the registration directly from the max-projections. Max-projections must have been computed before.

        """

        # First we pre-compute the max-projections and keep them in memory or save them on disk and load them up.
        projs = self._load_max_projs(path=path)

        # Then we loop again through the tiles but now we have access to the max-proj
        for tile in tqdm(self.tiles, desc='Compute cross-correlation'):
            proj1 = projs[tile.row, tile.col]

            for coords in tile.neighbors:
                proj2 = projs[coords[0], coords[1]]

                if tile.row == coords[0] and tile.col < coords[1]:
                    # EAST
                    if self.mask:
                        reg, rel = _get_masked_proj_shifts(proj1['east'], proj2['west'], threshold=self.threshold)
                    else:
                        reg, rel = _get_proj_shifts(proj1['east'], proj2['west'])

                elif tile.col == coords[1] and tile.row < coords[0]:
                    # SOUTH
                    if self.mask:
                        reg, rel = _get_masked_proj_shifts(proj1['south'], proj2['north'], threshold=self.threshold)
                    else:
                        reg, rel = _get_proj_shifts(proj1['south'], proj2['north'])

                else:
                    raise TypeError('Error: couldn''t determine registration to perform.')

                self.cgraph_from.append(np.ravel_multi_index([tile.row, tile.col],
                                                                    dims=(self.nrow, self.ncol)))
                self.cgraph_to.append(np.ravel_multi_index([coords[0], coords[1]],
                                                                  dims=(self.nrow, self.ncol)))

                # Regularize in cas of aberrant displacements
                reg, rel = self._regularize(reg, rel)

                # H=x, V=y, D=z
                self.dH.append(reg[2])
                self.dV.append(reg[1])
                self.dD.append(reg[0])
                self.relia_H.append(rel[2])
                self.relia_V.append(rel[1])
                self.relia_D.append(rel[0])

        self._build_sparse_graphs()
        self._optimize_sparse_graphs()
        _, _ = self._produce_registration_map()
        self._build_database()
        self._print_info()

    def compute_expected_registration(self):
        """
        Compute the expected registration if the expected overlap are correct.

        """

        reg_rel_map = np.zeros((3, self.nrow, self.ncol))

        self.registration_map_rel = reg_rel_map

        reg_abs_map = np.zeros_like(reg_rel_map)
        # H
        for x in range(reg_abs_map.shape[2]):
            reg_abs_map[0, :, x] = reg_rel_map[0, :, x] + x * (self.frame_size - self.expected_overlap_h)
        # V
        for x in range(reg_abs_map.shape[1]):
            reg_abs_map[1, x, :] = reg_rel_map[1, x, :] + x * (self.frame_size - self.expected_overlap_v)
        # D
        reg_abs_map[2] = reg_rel_map[2]
        self.registration_map_abs = reg_abs_map

        self._build_database()

    def plot_graph(self, annotate=False):
        """
        Plot the graph for each direction (H, D, V). This method needs to be called after the graph
        optimization.

        Parameters
        ----------
        annotate: bool
            control if annotation are drawn on the graph

        """

        if self.graph_relia_H is None:
            raise TypeError('Error: graph not build yet, please use build_sparse_graph()'
                            'before trying to plot the graph.')

        fig, ax = plt.subplots(1, 3)
        for i, d in enumerate(['H', 'V', 'D']):
            ind_from = getattr(self, 'cgraph_from')
            row, col = np.unravel_index(ind_from, shape=(self.nrow, self.ncol))
            V1 = np.vstack((row, col)).T

            ind_to = getattr(self, 'cgraph_to')
            row, col = np.unravel_index(ind_to, shape=(self.nrow, self.ncol))
            V2 = np.vstack((row, col)).T

            rel = getattr(self, 'relia_' + d)
            dX = getattr(self, 'd' + d)
            for ii in range(V1.shape[0]):
                ax[i].plot([V1[ii, 1], V2[ii, 1]], [V1[ii, 0], V2[ii, 0]], 'ko', markerfacecolor='r')
                if annotate:
                    p1 = ax[i].transData.transform_point([V1[ii, 1], V1[ii, 0]])
                    p2 = ax[i].transData.transform_point([V2[ii, 1], V2[ii, 0]])
                    dy = p2[1]-p1[1]
                    dx = p2[0]-p1[0]
                    rot = np.degrees(np.arctan2(dy, dx))
                    if rel[ii] < 0.15:
                        color = 'g'
                    elif rel[ii] < 0.30:
                        color = 'orange'
                    else:
                        color = 'r'
                    ax[i].annotate(text='err={:.2f} d{}={:.2f}'.format(rel[ii], d, dX[ii]),
                                   xy=((V1[ii, 1]+V2[ii, 1])/2, (V1[ii, 0]+V2[ii, 0])/2),
                                   ha='center',
                                   va='center',
                                   fontsize=8,
                                   rotation=rot,
                                   backgroundcolor='w',
                                   color=color)
            ax[i].set_title(d + ' tree')
            ax[i].invert_yaxis()

        return fig, ax

    def plot_min_trees(self, annotate=False):
        """
        Plot the minimum spanning tree for each direction (H, D, V). This method needs to be called after the graph
        optimization.

        Parameters
        ----------
        annotate: bool
            control if annotation are drawn on the graph

        """

        if self.min_tree_H is None:
            raise TypeError('Error: minimum spanning tree not computed yet, please use optimize_sparse_graph()'
                            'before trying to plot the trees.')

        fig, ax = self.plot_graph(annotate=False)

        for i, d in enumerate(['H', 'V', 'D']):
            ind_from = getattr(self, 'ctree_from_' + d)
            row, col = np.unravel_index(ind_from, shape=(self.nrow, self.ncol))
            V1 = np.vstack((row, col)).T

            ind_to = getattr(self, 'ctree_to_' + d)
            row, col = np.unravel_index(ind_to, shape=(self.nrow, self.ncol))
            V2 = np.vstack((row, col)).T

            dX = getattr(self, 'd' + d)
            for ii in range(V1.shape[0]):
                ax[i].plot([V1[ii, 1], V2[ii, 1]], [V1[ii, 0], V2[ii, 0]], 'ko-', markerfacecolor='r', linewidth=2)
                if annotate:
                    p1 = ax[i].transData.transform_point([V1[ii, 1], V1[ii, 0]])
                    p2 = ax[i].transData.transform_point([V2[ii, 1], V2[ii, 0]])
                    dy = p2[1]-p1[1]
                    dx = p2[0]-p1[0]
                    rot = np.degrees(np.arctan2(dy, dx))
                    ax[i].annotate(text='{:.2f}'.format(dX[self._get_ind(ind_from[ii], ind_to[ii])]),
                                   xy=((V1[ii, 1]+V2[ii, 1])/2, (V1[ii, 0]+V2[ii, 0])/2),
                                   ha='center',
                                   va='center',
                                   fontsize=8,
                                   rotation=rot,
                                   backgroundcolor='w',
                                   color='r')
            ax[i].set_title(d + ' tree')

    def plot_stitching_info(self):
        """
        Plot pair-wise registration error for each axis [H, V, D].

        Returns
        -------
        rel_map: array
            error matrix
        d_map: array
            shift matrix
        """

        if self.min_tree_H is None:
            raise TypeError('Error: minimum spanning tree not computed yet, please use optimize_sparse_graph()'
                            'before trying to plot stitching info.')

        rel_map = np.zeros((3, self.nrow, self.ncol))
        for i, d in enumerate(['H', 'V', 'D']):
            ind_from = getattr(self, 'ctree_from_' + d)
            ind_to = getattr(self, 'ctree_to_' + d)
            graph = getattr(self, 'graph_relia_' + d)
            rows, cols = np.unravel_index(ind_to, shape=(self.nrow, self.ncol))
            for row, col, i1, i2 in zip(rows, cols, ind_from, ind_to):
                rel = graph[i1, i2]
                rel_map[i, row, col] = np.max((rel_map[i, row, col], rel))
            rows, cols = np.unravel_index(ind_from, shape=(self.nrow, self.ncol))
            for row, col, i1, i2 in zip(rows, cols, ind_from, ind_to):
                rel = graph[i1, i2]
                rel_map[i, row, col] = np.max((rel_map[i, row, col], rel))

        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        for i, d in enumerate(['H', 'V', 'D']):
            h = ax[i].imshow(rel_map[i], cmap='turbo', vmin=0, vmax=2)
            ax[i].set_title('Registration {}'.format(d))
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, label='Estimated error [a.u.]')


        plt.figure()
        plt.imshow(np.mean(rel_map, axis=0), cmap='turbo')
        plt.colorbar(label='Total stimated error [a.u.]')

        if self.graph_relia_H is None:
            raise TypeError('Error: graph not build yet, please use build_sparse_graph()'
                            'before trying to plot the graph.')

        fig, ax = plt.subplots(1, 3)
        for i, d in enumerate(['H', 'V', 'D']):


            ind_from = getattr(self, 'cgraph_from')
            row, col = np.unravel_index(ind_from, shape=(self.nrow, self.ncol))
            V1 = np.vstack((row, col)).T+0.25

            ind_to = getattr(self, 'cgraph_to')
            row, col = np.unravel_index(ind_to, shape=(self.nrow, self.ncol))
            V2 = np.vstack((row, col)).T+0.25

            for ii in range(V1.shape[0]):
                ax[i].plot([V1[ii, 1], V2[ii, 1]], [V1[ii, 0], V2[ii, 0]], 'ko', markerfacecolor='r')
            ax[i].set_title(d + ' tree')
            ax[i].invert_yaxis()

            ind_from = getattr(self, 'ctree_from_' + d)
            row, col = np.unravel_index(ind_from, shape=(self.nrow, self.ncol))
            V1 = np.vstack((row, col)).T+0.25

            ind_to = getattr(self, 'ctree_to_' + d)
            row, col = np.unravel_index(ind_to, shape=(self.nrow, self.ncol))
            V2 = np.vstack((row, col)).T+0.25

            dX = getattr(self, 'd' + d)
            for ii in range(V1.shape[0]):
                ax[i].plot([V1[ii, 1], V2[ii, 1]], [V1[ii, 0], V2[ii, 0]], 'ko-', markerfacecolor='r', linewidth=2)
                p1 = ax[i].transData.transform_point([V1[ii, 1], V1[ii, 0]])
                p2 = ax[i].transData.transform_point([V2[ii, 1], V2[ii, 0]])
                dy = p2[1] - p1[1]
                dx = p2[0] - p1[0]
                rot = np.degrees(np.arctan2(dy, dx))
                ax[i].annotate(text='{:.2f}'.format(dX[self._get_ind(ind_from[ii], ind_to[ii])]),
                               xy=((V1[ii, 1] + V2[ii, 1]) / 2, (V1[ii, 0] + V2[ii, 0]) / 2),
                               ha='center',
                               va='center',
                               fontsize=8,
                               rotation=rot,
                               backgroundcolor='w',
                               color='r')

            sns.heatmap(self.registration_map_rel[i], annot=True, fmt='4.0f', ax=ax[i], cbar=False)

    def plot_registration_map(self):
        """
        Display the registration map using matplotlib.

        """

        if self.registration_map_abs is None:
            raise TypeError('Error: registration map not computed yet, please use produce_registration_map()'
                            'before trying to display the registration map.')

        fig, ax = plt.subplots(2, 3)
        for i, d in enumerate(['H', 'V', 'D']):
            ax[0, i].imshow(self.registration_map_rel[i], cmap='gray')
            ax[0, i].set_title('Rel reg. map ' + d)
            ax[1, i].imshow(self.registration_map_abs[i], cmap='gray')
            ax[1, i].set_title('Abs reg. map ' + d)

    def dump_stitcher(self, path):
        """
        Use dill to store a tgraph object.

        Parameters
        ----------
        path: string
            path to save the database.

        """
        if path[-4:] != '.pkl':
            path = path + '.pkl'

        with open(path, 'wb') as f:
            dill.dump(self, f)

    def set_overlap_margin(self, margin):
        """
        Modify the overlaping area size. If the overlaping area is smaller than the true one, the stitching can't
        be performed properly. If the overlaping area area is more than twice the size of the true one it will also
        fail (due to the circular FFT in the phase cross correlation).

        Parameters
        ----------
        margin: float
            safety margin in % to take the overlaping area.

        Returns
        -------
        None
        """
        if margin > 45:
            raise ValueError('Error: overlap margin is too big and will make the stitching fail.')
        if margin < 1:
            raise ValueError('Error: overlap margin is too small and may make the stitching fail.')

        self.overlap_h = int(self.expected_overlap_h*(1+margin/100))
        if self.expected_overlap_h > self.frame_size:
            self.expected_overlap_h = self.frame_size
        self.overlap_v = int(self.expected_overlap_v*(1+margin/100))
        if self.expected_overlap_v > self.frame_size:
            self.expected_overlap_v = self.frame_size

    def set_z_range(self, z_begin, z_end):
        """
        Set a range of depth fo computing the stitching.


        Parameters
        ----------
        z_begin: int
            first depth to be included in the max-proj
        z_end: int
            last depth to be included in the max-proj

        Returns
        -------
        None
        """

        self.z_begin = z_begin
        self.z_end = z_end

    def _print_info(self):
        """
        Display stitching result information.

        """
        overlap = np.median(np.diff(np.median(self.registration_map_abs[0], axis=0)))
        self.effective_overlap_h = (self.frame_size-overlap)/self.frame_size*100
        print('Effective horizontal overlap: {:0.2f}%'.format(self.effective_overlap_h))
        overlap = np.median(np.diff(np.median(self.registration_map_abs[1], axis=1)))
        self.effective_overlap_v = (self.frame_size-overlap)/self.frame_size*100
        print('Effective vertical overlap: {:0.2f}%'.format(self.effective_overlap_v))

        if np.abs(self.effective_overlap_v*self.frame_size/100-self.expected_overlap_v)>0.2*self.expected_overlap_v:
            warnings.warn('Expected vertical overlap is very different from the computed one, the registration '
                          'might be wrong.')
        if np.abs(self.effective_overlap_h*self.frame_size/100-self.expected_overlap_h)>0.2*self.expected_overlap_h:
            warnings.warn('Expected horizontal overlap is very different from the computed one, the registration '
                          'might be wrong.')

    def _build_sparse_graphs(self):
        """
        Build the sparse graph from the reliability and (row, col). This method needs to be called after the
        pair-wise registration has been performed for all neighbors pair.

        Returns
        -------
        None
        """

        csr_matrix_size = self.ncol*self.nrow
        self.graph_relia_H = csr_matrix((self.relia_H, (self.cgraph_from, self.cgraph_to)),
                                        shape=(csr_matrix_size, csr_matrix_size))
        self.graph_relia_V = csr_matrix((self.relia_V, (self.cgraph_from, self.cgraph_to)),
                                        shape=(csr_matrix_size, csr_matrix_size))
        self.graph_relia_D = csr_matrix((self.relia_D, (self.cgraph_from, self.cgraph_to)),
                                        shape=(csr_matrix_size, csr_matrix_size))

    def _optimize_sparse_graphs(self):
        """
        Optimize the sparse graph by computing the minimum spanning tree for each direction (H, D, V). This
        method needs to be called after the sparse graphs have been built.

        Returns
        -------
        None
        """

        if self.graph_relia_H is None:
            raise TypeError('Error: sparse graph not build yet, please use build_sparse_graph() before trying to'
                            'perform the optimization.')

        for g in ['graph_relia_H', 'graph_relia_V', 'graph_relia_D']:
            graph = getattr(self, g)
            # Minimum spanning tree
            min_tree = minimum_spanning_tree(graph)

            # Get the "true" neighbors
            min_tree = min_tree.tocoo()
            setattr(self, 'min_tree_' + g[-1], min_tree)
            ctree_from = min_tree.row
            setattr(self, 'ctree_from_' + g[-1], ctree_from)

            ctree_to = min_tree.col
            setattr(self, 'ctree_to_' + g[-1], ctree_to)

    def _produce_registration_map(self):
        """
        Produce the registration map where reg_rel_map[d, row, col] (d = H,V,D) is the relative tile
        position in pixel from the expected one. This method needs to be called after the optimization has been done.

        Returns
        -------
        None
        """

        if self.min_tree_H is None:
            raise TypeError('Error: minimum spanning tree not computed yet, please use optimize_sparse_graph()'
                            'before trying to compute the registration map.')

        # Relative registration
        # Initialize relative registration map
        reg_rel_map = np.zeros((3, self.nrow, self.ncol)) # H, V, D

        for i, min_tree in enumerate(['min_tree_H', 'min_tree_V', 'min_tree_D']):
            # Fill it by following the tree and getting the corresponding registration parameters
            node_array = depth_first_order(getattr(self, min_tree), i_start=self.cgraph_from[0],
                                           directed=False, return_predecessors=False)

            node_visited = [node_array[0]]

            tree = getattr(self, min_tree)
            row = tree.row
            col = tree.col

            for node_to in zip(node_array[1:]):
                # The previous node in the MST is a visited node with an edge to the current node
                neighbors = []
                for r, c in zip(row, col):
                    if r == node_to:
                        neighbors.append(c)
                    if c == node_to:
                        neighbors.append(r)
                node_from = [x for x in neighbors if x in node_visited]
                node_visited.append(node_to)

                # Get the previous neighbor local reg parameter
                ind1, ind2 = np.unravel_index(node_from, shape=(self.nrow, self.ncol))
                d_neighbor = reg_rel_map[i, ind1, ind2]

                # Get the current 2D tile position
                ind1, ind2 = np.unravel_index(node_to, shape=(self.nrow, self.ncol))
                # Get the associated ind position in the registration graph (as opposed to the reliability min_tree)
                ind_graph = self._get_ind(node_from, node_to)
                # Get the corresponding reg parameter
                d = getattr(self, 'd' + min_tree[-1])[ind_graph]
                # Get the corresponding relia and print a warning if it was regularized:
                relia = getattr(self, 'relia_' + min_tree[-1])[ind_graph]
                if relia == 2:
                    print('Aberrant pair-wise registration remaining after global optimization between tile ({},{}) '
                          'and tile ({},{})'.format(*np.unravel_index(node_from, shape=(self.nrow, self.ncol)),
                                                    *np.unravel_index(node_to, shape=(self.nrow, self.ncol))))
                # Update the local reg parameter in the 2D matrix
                if node_to > node_from[0]:
                    reg_rel_map[i, ind1, ind2] = d_neighbor + d
                else:
                    reg_rel_map[i, ind1, ind2] = d_neighbor - d
        self.registration_map_rel = reg_rel_map

        reg_abs_map = np.zeros_like(reg_rel_map)
        # H
        for x in range(reg_abs_map.shape[2]):
            reg_abs_map[0, :, x] = reg_rel_map[0, :, x] + x * (self.frame_size-self.overlap_h)
        # V
        for x in range(reg_abs_map.shape[1]):
            reg_abs_map[1, x, :] = reg_rel_map[1, x, :] + x * (self.frame_size-self.overlap_v)
        # D
        reg_abs_map[2] = reg_rel_map[2]
        self.registration_map_abs = reg_abs_map

        return reg_rel_map, reg_abs_map

    def _build_database(self):
        """
        Build the database for storing the registration parameters. This method needs to be called after
        the registration map has been produced.

        Returns
        -------
        None
        """

        if self.registration_map_rel is None:
            raise TypeError('Error: database can''t be build if the registration map has not been computed.'
                            ' Please use produce_registration_map() method first.')

        database_dict = {}
        for i in range(self.n_vertex):
            row = self.tiles[i].row
            col = self.tiles[i].col
            database_dict[i] = {'path': self.tiles[i].path,
                                'row': row,
                                'col': col,
                                'dH': self.registration_map_rel[0, row, col],
                                'dV': self.registration_map_rel[1, row, col],
                                'dD': self.registration_map_rel[2, row, col],
                                'ABS_H': self.registration_map_abs[0, row, col],
                                'ABS_V': self.registration_map_abs[1, row, col],
                                'ABS_D': self.registration_map_abs[2, row, col]}

        self.database = pd.DataFrame.from_dict(database_dict, orient='index')

        # Finally set the origin so that tile on the edge have coordinate 0 (rather than negative):
        for i, d in enumerate(['ABS_D', 'ABS_V', 'ABS_H']):
            self.database[d] = self.database[d] - self.database[d].min()

    def _get_ind(self, ind_from, ind_to):
        """
        Returns the ind in the original graph which corresponds to (ind_from, ind_to) in the minimum spanning tree.

        Parameters
        ----------
        ind_from: int
            starting node in the directed graph
        ind_to: int
            ending node in the directed graph

        Returns
        ----------
        ind: int
            corresponding ind in the original graph
        """
        ind = None
        for i, f in enumerate(self.cgraph_from):
            if f == ind_from:
                if self.cgraph_to[i] == ind_to:
                    ind = i
        if ind is None:
            for i, f in enumerate(self.cgraph_to):
                if f == ind_from:
                    if self.cgraph_from[i] == ind_to:
                        ind = i
        if ind is None:
            raise ValueError('Error: can''t find matching vertex pair.')
        return ind

    def _compute_east_registration(self, apr_1, parts_1, apr_2, parts_2):
        """
        Compute the registration between the current tile and its eastern neighbor.

        Parameters
        ----------
        u: list
            current tile
        v: list
            neighboring tile

        Returns
        -------
        None
        """
        patch = pyapr.ReconPatch()
        patch.y_begin = self.frame_size - self.overlap_h
        proj_zy1, proj_zx1, proj_yx1 = _get_max_proj_apr(apr_1, parts_1, patch, plot=False)

        patch = pyapr.ReconPatch()
        patch.y_end = self.overlap_h
        proj_zy2, proj_zx2, proj_yx2 = _get_max_proj_apr(apr_2, parts_2, patch, plot=False)

        # proj1, proj2 = [proj_zy1, proj_zx1, proj_yx1], [proj_zy2, proj_zx2, proj_yx2]
        # for i, title in enumerate(['X', 'Y', 'Z']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('EAST')
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)

        if self.mask:
            return _get_masked_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                           [proj_zy2, proj_zx2, proj_yx2],
                                           threshold=self.threshold)
        else:
            return _get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                    [proj_zy2, proj_zx2, proj_yx2])

    def _compute_south_registration(self, apr_1, parts_1, apr_2, parts_2):
        """
        Compute the registration between the current tile and its southern neighbor.

        Parameters
        ----------
        u: list
            current tile
        v: list
            neighboring tile

        Returns
        -------
        None
        """
        patch = pyapr.ReconPatch()
        patch.x_begin = self.frame_size - self.overlap_v
        proj_zy1, proj_zx1, proj_yx1 = _get_max_proj_apr(apr_1, parts_1, patch, plot=False)

        patch = pyapr.ReconPatch()
        patch.x_end = self.overlap_v
        proj_zy2, proj_zx2, proj_yx2 = _get_max_proj_apr(apr_2, parts_2, patch, plot=False)

        # proj1, proj2 = [proj_zy1, proj_zx1, proj_yx1], [proj_zy2, proj_zx2, proj_yx2]
        # for i, title in enumerate(['X', 'Y', 'Z']):
        #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        #     ax[0].imshow(proj1[i], cmap='gray')
        #     ax[0].set_title('SOUTH')
        #     ax[1].imshow(proj2[i], cmap='gray')
        #     ax[1].set_title(title)

        if self.mask:
            return _get_masked_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                           [proj_zy2, proj_zx2, proj_yx2],
                                           threshold=self.threshold)
        else:
            return _get_proj_shifts([proj_zy1, proj_zx1, proj_yx1],
                                    [proj_zy2, proj_zx2, proj_yx2])


class channelStitcher(baseStitcher):
    """
    Class used to perform the stitching between different channels. The registration must be performed first a single
    channel (typically auto-fluorescence)

    The stitching is performed between each corresponding tile and the relative displacement is added to the previously
    determined stitching parameters.

    The number and position of tile should matched for bot dataset.
    """

    def __init__(self,
                 stitcher,
                 ref,
                 moving):
        """
        Constructor for the channelStitcher class.

        Parameters
        ----------
        stitcher: tileStitcher
            tileStitcher object with the multitile registration parameters
        tiles_stitched: tileParser
            tiles corresponding to the stitcher
        tiles_channel: tileParser
            tiles to be registered to tiles_stitched
        """

        super().__init__(moving,
                         stitcher.overlap_h,
                         stitcher.overlap_v)

        self.stitcher = stitcher
        self.tiles_ref = ref
        self.database = stitcher.database.copy()
        # Change tiles path for the channel tiles
        self.database.path = self.tiles.path_list

        self.segment = False
        self.segmentation_verbose = None

        self.patch = pyapr.ReconPatch()

    def compute_rigid_registration(self, progress_bar=True):
        """
        Compute the rigid registration between each pair of tiles across different channels.

        Returns
        -------
        None
        """

        for tile1, tile2 in zip(tqdm(self.tiles_ref, desc='Computing rigid registration', disable=not progress_bar),
                                self.tiles):

            tile1.load_tile()
            tile2.load_tile()

            if self.segment:
                self.segmenter.compute_segmentation(tile2)

            proj1 = _get_max_proj_apr(tile1.apr, tile1.parts, self.patch)
            proj2 = _get_max_proj_apr(tile2.apr, tile2.parts, self.patch)

            if self.mask:
                reg, rel = _get_masked_proj_shifts(proj1, proj2, self.threshold)
            else:
                reg, rel = _get_proj_shifts(proj1, proj2)

            # TODO: add regularization to avoid aberrant shifts.

            self._update_database(tile2.row, tile2.col, reg)

    def set_lim(self, x_begin=None, x_end=None, y_begin=None, y_end=None, z_begin=None, z_end=None):
        """
        Define spatial limits to compute the maximum intensity projection.

        Parameters
        ----------
        x_begin: int
        x_end: int
        y_begin: int
        y_end: int
        z_begin: int
        z_end: int

        Returns
        -------
        None
        """

        if x_begin is not None:
            self.patch.x_begin = x_begin
        if x_end is not None:
            self.patch.x_end = x_end
        if y_begin is not None:
            self.patch.y_begin = y_begin
        if y_end is not None:
            self.patch.y_end = y_end
        if z_begin is not None:
            self.patch.z_begin = z_begin
        if z_end is not None:
            self.patch.z_end = z_end

    def _update_database(self, row, col, d):
        """
        Update database after the registration.

        Parameters
        ----------
        row: int
            row number
        col: int
            col number
        d: int
            computed displacement

        Returns
        -------

        """
        d = np.concatenate([d, d])
        df = self.database
        for loc, value in zip(['dD', 'dV', 'dH', 'ABS_D', 'ABS_V', 'ABS_H'], d):
            df.loc[(df['row'] == row) & (df['col'] == col), loc] += value


class tileMerger():
    """
    Class to merge tiles and create a stitched volume. Typically used at a lower resolution for registering
    the sample to an Atlas.

    """
    def     __init__(self, tiles, database):
        """
        Constructor for the tileMerger class.

        Parameters
        ----------
        tiles:  tileParser
            tileParser object containing the dataset to merge.
        database: (pd.DataFrame, string)
            database or path to the database containing the registered tile position
        n_planes: int
            number of planes per files.

        Returns
        -------
        None
        """
        if isinstance(database, str):
            self.database = pd.read_csv(database)
        else:
            self.database = database
        self.tiles = tiles
        self.lazy = self._find_if_lazy()
        self.type = tiles.type
        self.frame_size = tiles.frame_size
        self.n_planes = self._get_n_planes()
        self.n_tiles = tiles.n_tiles
        self.n_row = tiles.nrow
        self.n_col = tiles.ncol

        # Size of the merged array (to be defined when the merged array is initialized).
        self.nx = None
        self.ny = None
        self.nz = None

        self.downsample = 1
        self.level_delta = 0
        self.merged_data = None
        self.merged_segmentation = None

    def merge_additive(self, reconstruction_mode='constant', tree_mode='mean', progress_bar=True):
        """
        Perform merging with a mean algorithm for overlapping areas. Maximum merging should be preferred to
        avoid integer overflowing and higher signals on the overlapping areas.

        Parameters
        ----------
        mode: string
            APR reconstruction type among ('constant', 'smooth', 'level')

        Returns
        -------
        None
        """
        if self.merged_data is None:
            self._initialize_merged_array()

        H_pos = self.database['ABS_H'].to_numpy()
        H_pos = (H_pos - H_pos.min())/self.downsample
        V_pos = self.database['ABS_V'].to_numpy()
        V_pos = (V_pos - V_pos.min())/self.downsample
        D_pos = self.database['ABS_D'].to_numpy()
        D_pos = (D_pos - D_pos.min())/self.downsample

        for i, tile in enumerate(tqdm(self.tiles, desc='Merging', disable=not progress_bar)):

            if self.type == 'apr':
                if self.lazy:
                    tile.lazy_load_tile(level_delta=self.level_delta)
                    data = tile.lazy_data[:, :, :]
                else:
                    tile.load_tile()
                    u = pyapr.reconstruction.APRSlicer(tile.apr, tile.parts, level_delta=self.level_delta,
                                                        mode=reconstruction_mode, tree_mode=tree_mode)
                    data = u[:, :, :]
            else:
                tile.load_tile()
                data = tile.data

            x1 = int(H_pos[i])
            x2 = int(H_pos[i] + data.shape[2])
            y1 = int(V_pos[i])
            y2 = int(V_pos[i] + data.shape[1])
            z1 = int(D_pos[i])
            z2 = int(D_pos[i] + data.shape[0])

            self.merged_data[z1:z2, y1:y2, x1:x2] = self.merged_data[z1:z2, y1:y2, x1:x2] + data
            self.merged_data = self.merged_data.astype('uint16')

    def merge_max(self, reconstruction_mode='constant', tree_mode='mean', debug=False, progress_bar=True):
        """
        Perform merging with a maximum algorithm for overlapping areas.

        Parameters
        ----------
        mode: string
            APR reconstruction type among ('constant', 'smooth', 'level')
        debug: bool
            add white border on the edge of each tile to see where it was overlapping.

        Returns
        -------
        None
        """
        if self.merged_data is None:
            self._initialize_merged_array()

        H_pos = self.database['ABS_H'].to_numpy()
        H_pos = (H_pos - H_pos.min())/self.downsample
        V_pos = self.database['ABS_V'].to_numpy()
        V_pos = (V_pos - V_pos.min())/self.downsample
        D_pos = self.database['ABS_D'].to_numpy()
        D_pos = (D_pos - D_pos.min())/self.downsample

        for i, tile in enumerate(tqdm(self.tiles, desc='Merging', disable=not progress_bar)):

            if self.type == 'apr':
                if self.lazy:
                    tile.lazy_load_tile(level_delta=self.level_delta)
                    data = tile.lazy_data[:, :, :]
                else:
                    tile.load_tile()
                    u = pyapr.reconstruction.APRSlicer(tile.apr, tile.parts, level_delta=self.level_delta,
                                                        mode=reconstruction_mode, tree_mode=tree_mode)
                    data = u[:, :, :]
            else:
                tile.load_tile()
                data = downscale_local_mean(tile.data, factors=(self.downsample, self.downsample, self.downsample))

            # In debug mode we highlight each tile edge to see where it was
            if debug:
                data[0, :, :] = 2**16-1
                data[-1, :, :] = 2 ** 16 - 1
                data[:, 0, :] = 2 ** 16 - 1
                data[:, -1, :] = 2 ** 16 - 1
                data[:, :, 0] = 2 ** 16 - 1
                data[:, :, -1] = 2 ** 16 - 1

            x1 = int(H_pos[i])
            x2 = int(H_pos[i] + data.shape[2])
            y1 = int(V_pos[i])
            y2 = int(V_pos[i] + data.shape[1])
            z1 = int(D_pos[i])
            z2 = int(D_pos[i] + data.shape[0])

            self.merged_data[z1:z2, y1:y2, x1:x2] = np.maximum(self.merged_data[z1:z2, y1:y2, x1:x2], data)

    def merge_segmentation(self, reconstruction_mode='constant', tree_mode='max', debug=False, progress_bar=True):
        """
        Perform merging with a maximum algorithm for overlapping areas.

        Parameters
        ----------
        mode: string
            APR reconstruction type among ('constant', 'smooth', 'level')
        debug: bool
            add white border on the edge of each tile to see where it was overlapping.

        Returns
        -------
        None
        """
        if self.merged_segmentation is None:
            self._initialize_merged_segmentation()

        H_pos = self.database['ABS_H'].to_numpy()
        H_pos = (H_pos - H_pos.min())/self.downsample
        V_pos = self.database['ABS_V'].to_numpy()
        V_pos = (V_pos - V_pos.min())/self.downsample
        D_pos = self.database['ABS_D'].to_numpy()
        D_pos = (D_pos - D_pos.min())/self.downsample

        for i, tile in enumerate(tqdm(self.tiles, desc='Merging', disable=not progress_bar)):

            if self.type == 'apr':
                if self.lazy:
                    tile.lazy_load_segmentation(level_delta=self.level_delta)
                    data = tile.lazy_segmentation[:, :, :]
                else:
                    tile.load_segmentation()
                    u = pyapr.reconstruction.APRSlicer(tile.apr, tile.parts_cc, level_delta=self.level_delta,
                                                        mode=reconstruction_mode, tree_mode=tree_mode)
                    data = u[:, :, :]

            # In debug mode we highlight each tile edge to see where it was
            if debug:
                data[0, :, :] = 2**16-1
                data[-1, :, :] = 2 ** 16 - 1
                data[:, 0, :] = 2 ** 16 - 1
                data[:, -1, :] = 2 ** 16 - 1
                data[:, :, 0] = 2 ** 16 - 1
                data[:, :, -1] = 2 ** 16 - 1

            x1 = int(H_pos[i])
            x2 = int(H_pos[i] + data.shape[2])
            y1 = int(V_pos[i])
            y2 = int(V_pos[i] + data.shape[1])
            z1 = int(D_pos[i])
            z2 = int(D_pos[i] + data.shape[0])

            self.merged_segmentation[z1:z2, y1:y2, x1:x2] = np.maximum(self.merged_segmentation[z1:z2, y1:y2, x1:x2], data)

    def crop(self, background=0, xlim=None, ylim=None, zlim=None):
        """
        Add a black mask around the brain (rather than really cropping which makes the overlays complicated in
        a later stage).

        Parameters
        ----------
        background: int
            constant value to replace the cropped area with.
        xlim: array_like
            x limits for cropping
        ylim: array_like
            y limits for cropping
        zlim: array_like
            z limits for cropping

        Returns
        -------
        None
        """

        if self.merged_data is None:
            raise TypeError('Error: please merge data before cropping.')

        if xlim is not None:
            if xlim[0] != 0:
                self.merged_data[:, :, :xlim[0]] = background
            if xlim[1] != self.merged_data.shape[2]:
                self.merged_data[:, :, xlim[1]:] = background
        if ylim is not None:
            if ylim[0] != 0:
                self.merged_data[:, :ylim[0], :] = background
            if ylim[1] != self.merged_data.shape[1]:
                self.merged_data[:, ylim[1]:, :] = background
        if zlim is not None:
            if zlim[0] != 0:
                self.merged_data[:zlim[0], :, :] = background
            if zlim[1] != self.merged_data.shape[0]:
                self.merged_data[zlim[1]:, :, :] = background

    def equalize_hist(self, method='opencv'):
        """
        Perform histogram equalization to improve the contrast on merged data.
        Both OpenCV (only 2D) and Skimage (3D but 10 times slower) are available.

        Parameters
        ----------
        method: string
            method for performing histogram equalization among 'skimage' and 'opencv'.

        Returns
        -------
        None
        """
        if self.merged_data is None:
            raise TypeError('Error: please merge data before equalizing histogram.')

        if method == 'opencv':
            clahe = cv.createCLAHE(tileGridSize=(8, 8))
            for i in range(self.merged_data.shape[0]):
                self.merged_data[i] = clahe.apply(self.merged_data[i])
        elif method == 'skimage':
            self.merged_data = equalize_adapthist(self.merged_data)
        else:
            raise ValueError('Error: unknown method for adaptive histogram normalization.')

    def set_downsample(self, downsample):
        """
        Set the downsampling value for the merging reconstruction.

        Parameters
        ----------
        downsample: int
            downsample factor

        Returns
        -------
        None

        """

        # TODO: find a more rigorous way of enforcing this. (Probably requires that the APR is loaded).
        if downsample not in [1, 2, 4, 8, 16, 32]:
            raise ValueError('Error: downsample value should be compatible with APR levels.')

        self.downsample = downsample
        self.level_delta = int(-np.log2(self.downsample))

    def _initialize_merged_array(self):
        """
        Initialize the merged array in accordance with the asked downsampling.

        Returns
        -------
        None
        """
        self.nx = int(np.ceil(self._get_nx() / self.downsample))
        self.ny = int(np.ceil(self._get_ny() / self.downsample))
        self.nz = int(np.ceil(self._get_nz() / self.downsample))

        self.merged_data = np.zeros((self.nz, self.ny, self.nx), dtype='uint16')

    def _initialize_merged_segmentation(self):
        """
        Initialize the merged array in accordance with the asked downsampling.

        Returns
        -------
        None
        """
        self.nx = int(np.ceil(self._get_nx() / self.downsample))
        self.ny = int(np.ceil(self._get_ny() / self.downsample))
        self.nz = int(np.ceil(self._get_nz() / self.downsample))

        self.merged_segmentation = np.zeros((self.nz, self.ny, self.nx), dtype='uint16')

    def _get_nx(self):
        """
        Compute the merged array size for x dimension.

        Returns
        -------
        _: int
            x size for merged array
        """
        x_pos = self.database['ABS_H'].to_numpy()
        return x_pos.max() - x_pos.min() + self.frame_size

    def _get_ny(self):
        """
        Compute the merged array size for y dimension.

        Returns
        -------
        _: int
            y size for merged array
        """
        y_pos = self.database['ABS_V'].to_numpy()
        return y_pos.max() - y_pos.min() + self.frame_size

    def _get_nz(self):
        """
        Compute the merged array size for y dimension.

        Returns
        -------
        _: int
            y size for merged array
        """
        z_pos = self.database['ABS_D'].to_numpy()
        return z_pos.max() - z_pos.min() + self.n_planes

    def _get_n_planes(self):
        """
        Load a tile and check the number of planes per tile.

        Returns
        -------
        _: int
            Number of planes per tile;
        """
        tile = self.tiles[0]
        if self.type == 'apr':
            if self.lazy:
                tile.lazy_load_tile()
                return tile.lazy_data.shape[0]
            else:
                tile.load_tile()
                return tile.apr.shape()[0]
        else:
            tile.load_tile()
            return tile.data.shape[0]

    def _find_if_lazy(self):
        """
        Function to test if all tile can be lazy loaded.

        Returns
        -------
        _: bool
            True if the tile can be lazy loaded, false if not.
        """
        try:
            for tile in self.tiles:
                tile.lazy_load_tile()
        except:
            return False

        return True
