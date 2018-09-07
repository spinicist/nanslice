#!/usr/bin/env python
"""
The nanslice package. Everything important is in a submodule, but the following
objects and functions are imported here for convenient access:

- :py:obj:`~nanslice.box.Box`
- :py:obj:`~nanslice.slicer.Slicer`
- :py:obj:`~nanslice.layer.Layer`, :py:func:`~nanslice.layer.blend_layers`

"""

from .box import Box
from .slicer import Slicer, axis_indices, axis_map
from .layer import Layer, blend_layers
