#!/usr/bin/env python
"""Box.py

A simple bounding-box class"""

import numpy as np

class Box:
    """A simple bounding-box class"""

    def __init__(self, img, mask=False):
        if mask:
            self._c = _mask_bbox(img)
        else:
            self._c = _full_bbox(img)
        self._diag = self._c[1] - self._c[0]
        self._center = (self._c[0] + self._c[1]) / 2

    def __str__(self):
        return 'Box Start: ' + str(self.start) + ' End: ' + str(self.end)

    @property
    def start(self):
        return self._c[0]

    @property
    def end(self):
        return self._c[1]

    @property
    def diag(self):
        """Returns the diagonal of the bounding-box"""
        return self._diag

    @property
    def center(self):
        """Returns the geometric center of the bounding-box"""
        return self._center

def _full_bbox(img):
    img_shape = img.get_data().shape
    corners = np.array([[0, 0, 0, 1.],
                        [img_shape[0], img_shape[1], img_shape[2], 1.]])
    corners = np.dot(img.get_affine(), corners.T)
    corner1 = np.min(corners[0:3, :], axis=1)
    corner2 = np.max(corners[0:3, :], axis=1)
    return corner1, corner2

def _mask_bbox(img, padding=0):
    """Finds the bounding box of non-zero voxels"""
    data = img.get_data()

    # Individual axis min/maxes
    xmin, xmax = np.where(np.any(data, axis=(1, 2)))[0][[0, -1]]
    ymin, ymax = np.where(np.any(data, axis=(0, 2)))[0][[0, -1]]
    zmin, zmax = np.where(np.any(data, axis=(0, 1)))[0][[0, -1]]

    # Convedir_rt to physical space
    corners = np.array([[xmin, ymin, zmin, 1.],
                        [xmax, ymax, zmax, 1.]])
    corners = np.dot(img.get_affine(), corners.T)
    # Now do min/maxes again to standardise corners
    corner1 = np.min(corners[0:3, :], axis=1) - padding
    corner2 = np.max(corners[0:3, :], axis=1) + padding
    return corner1, corner2
