#!/usr/bin/env python
"""nanslice

Not Another Neuroimaging Slicer - A Package for creating beautiful neuroimages in Python"""

from .box import Box
from .slicer import Slicer, axis_indices, axis_map
from .layer import Layer, blend_layers
from .jupyter import three_plane, slices
from .util import colorbar, alphabar