#!/usr/bin/env python
"""Slice.py

The core Slice object"""
import numpy as np
import scipy.ndimage.interpolation as ndinterp

axis_map = {'x':0, 'y':1, 'z':2}
orient_map = {'clin': ({0: 1, 1: 0, 2: 0}, {0: 2, 1: 2, 2: 1}),
              'preclin': ({0: 2, 1: 0, 2: 0}, {0: 1, 1: 2, 2: 1})}
def axis_indices(slice_index, orient='clin'):
    this_orient = orient_map[orient]
    return (this_orient[0][slice_index], this_orient[1][slice_index])

class Slice:
    """A very simple slice class. Stores physical & voxel space co-ords"""
    def __init__(self, bbox, axis, pos, samples=64,
                 absolute=False, orient='clin'):
        ind_0 = axis_map[axis]
        ind_1, ind_2 = axis_indices(ind_0, orient=orient)
        start = np.copy(bbox.start)
        if absolute:
            start[ind_0] = pos
        else:
            start[ind_0] = bbox.start[ind_0]*(1-pos) + bbox.end[ind_0]*pos
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
        """Samples an image using this slice"""
        physical = self.get_physical(img.affine)
        return ndinterp.map_coordinates(img.get_data().squeeze(), physical, order=order).T