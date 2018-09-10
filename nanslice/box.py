#!/usr/bin/env python
"""Box.py

Contains a simple bounding-box class"""

import numpy as np

class Box:
    """
    A simple bounding-box class
    
    Constructor parameters:

    - center -- The center of the box (x,y,z) in world-space co-ordinates
    - size   -- The size of the box (x,y,z) in world-space units
    - corners -- Two corners (x,y,z) of the box in world-space co-ordinates

    Either corners or both center and size must be specified
    """
    def __init__(self, center=None, size=None, corners=None):
        if center is not None and size is not None:
            self._center = center
            self._diag = size
            self._c = (center - size/2, center + size/2)
        elif corners is not None:
            self._c = corners
            self._diag = self._c[1] - self._c[0]
            self._center = (self._c[0] + self._c[1]) / 2
        else:
            raise Exception('Either center & sz or corners must be specified')

    def __str__(self):
        return 'Box Start: ' + str(self.start) + ' End: ' + str(self.end)

    @classmethod
    def fromImage(cls, img):
        """
        Creates a bounding box from the corners defined by an image
        
        Parameters:

        - img -- An nibabel image
        """
        img_shape = img.get_data().shape
        corners = np.array([[0, 0, 0, 1.],
                            [img_shape[0], img_shape[1], img_shape[2], 1.]])
        corners = np.dot(img.get_affine(), corners.T)
        corner1 = np.min(corners[0:3, :], axis=1)
        corner2 = np.max(corners[0:3, :], axis=1)
        return cls(corners=(corner1, corner2))
    
    @classmethod
    def fromMask(cls, img, padding=0):
        """
        Creates a bounding box that encloses all non-zero voxels in a volume
        
        Parameters:
        
        - img -- The volume to create the bounding-box from
        - padding -- Number of extra voxels to pad the resulting box by
        """
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
        return cls(corners=(corner1, corner2))

    @property
    def start(self):
        """Returns the 'start' corner of the bounding-box"""
        return self._c[0]

    @property
    def end(self):
        """Returns the 'end' corner of the bounding-box (opposite the start)"""
        return self._c[1]

    @property
    def diag(self):
        """Returns the vector diagonal of the bounding-box"""
        return self._diag

    @property
    def center(self):
        """Returns the geometric center of the bounding-box"""
        return self._center

    def slice_positions(self, num_slices, start=0, end=1):
        """Returns an array of slice positions along the specified axis"""
        slice_pos = self.start + self.diag * np.linspace(start, end, num_slices)[:, np.newaxis]
        return slice_pos
