#!/usr/bin/env python
"""
slicer.py

This module contains the core Slicer object that samples Layers to produce
image arrays that can be drawn with matlplotlib.
"""
import numpy as np
import scipy.ndimage.interpolation as ndinterp

Axis_map = {'x': 0, 'y': 1, 'z': 2}
Orient_map = {'clin': ({0: 1, 1: 0, 2: 0}, {0: 2, 1: 2, 2: 1}),
              'preclin': ({0: 2, 1: 2, 2: 0}, {0: 1, 1: 0, 2: 1})}


def axis_indices(axis, orient='clin'):
    """
    Returns a pair of indices corresponding to right/up for the given orientation.
    Parameters:
    axis:   The perpendicular axis to the slice. Use Axis_map to convert between x/y/z and 0/1/2
    orient: Either 'clin' or 'preclin'
    """
    this_orient = Orient_map[orient]
    return (this_orient[0][axis], this_orient[1][axis])


class Slicer:
    """
    The Slicer class.

    When constructed, creates an array of world-space co-ordinates which represent the desired slice

    Constructor Parameters:

    - bbox -- :py:class:`~nanslice.box.Box` instance that defines the world-space bounding box that you want to slice
    - pos -- Position within the box to generate the slice through
    - axis -- Which axis you want to slice across. Either x/y/z or 0/1/2
    - samples -- Number of samples in the horizontal direction
    - orient --  'clin' or 'preclin'
    """

    def __init__(self, bbox, pos, axis, samples=64, orient='clin'):
        try:
            ind_0 = Axis_map[axis]  # If someone passed in x/y/z
        except KeyError:
            ind_0 = axis  # Assume it was an integer

        ind_1, ind_2 = axis_indices(ind_0, orient=orient)
        start = np.copy(bbox.start)
        start[ind_0] = pos
        dir_rt = np.zeros((3,))
        dir_up = np.zeros((3,))
        dir_rt[ind_1] = bbox.diag[ind_1]
        dir_up[ind_2] = bbox.diag[ind_2]
        aspect = np.linalg.norm(dir_up) / np.linalg.norm(dir_rt)
        samples_up = int(round(aspect * samples))
        self._world_space = (start[:, None, None] +
                             dir_rt[:, None, None] * np.linspace(0, 1, samples)[None, :, None] +
                             dir_up[:, None, None] * np.linspace(0, 1, samples_up)[None, None, :])
        # This is the extent parameter for matplotlib
        self.extent = (bbox.start[ind_1], bbox.end[ind_1],
                       bbox.start[ind_2], bbox.end[ind_2])
        self._tfm = None
        self._voxel_space = None

    def get_voxel_coords(self, tfm):
        """
        Returns an array of voxel space co-ordinates for this slice, which will be cached.
        If a subsequent call uses the same affine transform, the cached co-ordinates will be returned.
        If a new transform is passed in, then a fresh set of co-ordinates are calculated first.

        Parameters:

        - tfm -- An affine transform that defines an images physical space (usually the .affine property of an nibabel image)
        """
        if not np.array_equal(tfm, self._tfm):
            old_sz = self._world_space.shape
            new_sz = np.prod(self._world_space.shape[1:])
            scale = np.mat(tfm[0:3, 0:3]).I
            offset = np.dot(-scale, tfm[0:3, 3]).T
            isl = np.dot(scale, self._world_space.reshape(
                [3, new_sz])) + offset[:]
            isl = np.array(isl).reshape(old_sz)
            self._voxel_space = isl
            self._tfm = tfm
        return self._voxel_space

    def sample(self, img_data, affine, order, scale=1.0, volume=0):
        """
        Samples the passed 3D/4D image and returns a 2D slice

        Paramters:

        - img -- An nibabel image
        - order -- Interpolation order. 1 is linear interpolation
        - scale -- Scale factor to multiply all voxel values by
        - volume -- If sampling 4D data, specify the desired volume

        """
        physical = self.get_voxel_coords(affine)
        # Support timeseries by adding an extra co-ord specifying the volume
        if len(img_data.shape) == 4:
            vol_index = np.tile(volume, physical.shape[1:3])[np.newaxis, :]
            physical = np.concatenate((physical, vol_index), axis=0)
        return scale * ndinterp.map_coordinates(img_data, physical, order=order).T
