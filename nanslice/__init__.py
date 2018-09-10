#!/usr/bin/env python
"""
The nanslice package. Everything important is in a submodule, but the
:py:class:`~nanslice.slicer.Slicer` and :py:class:`~nanslice.layer.Layer` classes,
and the :py:func:`~nanslice.layer.blend_layers` function, are imported here
for convenient access. Between them, they do all the important work.
"""

from .layer import Layer, blend_layers
from .slicer import Slicer
