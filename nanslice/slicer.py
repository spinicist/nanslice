#!/usr/bin/env python
"""slicer.py

This module contains the core Slicer object that samples images to produce
slices."""
import numpy as np
import scipy.ndimage.interpolation as ndinterp

axis_map = {'x':0, 'y':1, 'z':2}
orient_map = {'clin': ({0: 1, 1: 0, 2: 0}, {0: 2, 1: 2, 2: 1}),
              'preclin': ({0: 2, 1: 2, 2: 0}, {0: 1, 1: 0, 2: 1})}
def axis_indices(axis, orient='clin'):
    """Returns a pair of indices corresponding to right/up for the given orientation.
    Parameters:
        axis:   The perpendicular axis to the slice. Use axis_map to convert between x/y/z and 0/1/2
        orient: Either 'clin' or 'preclin'
    """
    this_orient = orient_map[orient]
    return (this_orient[0][axis], this_orient[1][axis])

class Slicer:
    """The Slicer class. When constructed, creates an array of physical space co-ords, which are
    used by sample() to sample a 3D volume.
    Constructor Parameters:
        bbox:    Bounding-Box that you want to slice
        pos:     Position within the box to generate the slice through
        axis:    Which axis you want to slice across
        samples: Number of samples in the 'right' direction
        orient:  'clin' or 'preclin'
    """

    def __init__(self, bbox, pos, axis, samples=64, orient='clin'):
        try:
            ind_0 = axis_map[axis] # If someone passed in x/y/z
        except:
            ind_0 = axis # Assume it was an integer
        
        ind_1, ind_2 = axis_indices(ind_0, orient=orient)
        start = np.copy(bbox.start)
        start[ind_0] = pos[ind_0]
        dir_rt = np.zeros((3,))
        dir_up = np.zeros((3,))
        dir_rt[ind_1] = bbox.diag[ind_1]
        dir_up[ind_2] = bbox.diag[ind_2]
        aspect = np.linalg.norm(dir_up) / np.linalg.norm(dir_rt)
        samples_up = np.round(aspect * samples)
        self._world_space = (start[:, None, None] +
                             dir_rt[:, None, None] * np.linspace(0, 1, samples)[None, :, None] +
                             dir_up[:, None, None] * np.linspace(0, 1, samples_up)[None, None, :])
        # This is the extent parameter for matplotlib
        self.extent = (bbox.start[ind_1], bbox.end[ind_1], bbox.start[ind_2], bbox.end[ind_2])
        self._tfm = None
        self._voxel_space = None

    def get_physical(self, tfm):
        """Returns an array of physical space co-ords for this slice. Will be cached."""
        if not np.array_equal(tfm, self._tfm):
            old_sz = self._world_space.shape
            new_sz = np.prod(self._world_space.shape[1:])
            scale = np.mat(tfm[0:3, 0:3]).I
            offset = np.dot(-scale, tfm[0:3, 3]).T
            isl = np.dot(scale, self._world_space.reshape([3, new_sz])) + offset[:]
            isl = np.array(isl).reshape(old_sz)
            self._voxel_space = isl
            self._tfm = tfm
        return self._voxel_space

    def sample(self, img, order):
        """Samples a volume using the calculated slice co-ordinates"""
        physical = self.get_physical(img.affine)
        return ndinterp.map_coordinates(img.get_data().squeeze(), physical, order=order).T