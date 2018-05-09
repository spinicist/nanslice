#!/usr/bin/env python
"""nanslice

Not Another Neuroimaging Slicer - A Package for creating beautiful neuroimages in Python"""

from . import image
from . import util

from .box import Box
from .slicer import Slicer, axis_indices, axis_map

from .jupyter import three_plane, slices