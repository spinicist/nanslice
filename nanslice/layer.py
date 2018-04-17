#!/usr/bin/env python
"""layer.py

Contains the Layer class which stores settings for each layer (image to sample,
cmap, transparency, etc., and the overlay function for combining layers
"""

from pathlib import Path
from numpy import nanpercentile, ones_like
from nibabel import load
from . import image

def check_path(maybe_path):
    """Helper function to check if an object is path-like"""
    if isinstance(maybe_path, Path) or isinstance(maybe_path, str):
        return True
    else:
        return False

class Layer:
    """The Layer class. Keeps tabs on the image, color-map & transparency value"""

    def __init__(self, image, scale=1.0,
                 cmap=None, clim=None,
                 mask=None, mask_threshold=None,
                 alpha=None, alpha_lims=None):
        if check_path(image):
            self.image = load(str(image))
        else:
            self.image = image
        self.scale = scale
        if cmap:
            self.cmap = cmap
        else:
            self.cmap = 'gist_gray'
        if clim:
            self.clim = clim
        else:
            self.clim = nanpercentile(self.image.get_data(), (2, 98))
        if check_path(mask):
            self.mask_image = load(str(mask))
        else:
            self.mask_image = mask
        if mask_threshold:
            self.mask_threshold = mask_threshold
        else:
            self.mask_threshold = None
        if check_path(alpha):
            self.alpha_image = load(str(alpha))
            if alpha_lims:
                self.alpha_lims = alpha_lims
            else:
                self.alpha_lims = nanpercentile(self.alpha_image.get_data(), (2, 98))
        elif alpha:
            self.alpha_image = None
            self.alpha = alpha
        else:
            self.alpha_image = None
            self.alpha = 1.0

def overlay(slicer, base, overlays, interp_order):
    """Blends together a set of overlays"""
    base_slice = image.colorize(slicer.sample(base.image, interp_order) * base.scale,
                                base.cmap, base.clim)
    if overlays:
        for over in overlays:
            over_slice = slicer.sample(over.image, interp_order) * over.scale
            if over.mask_threshold:
                over_slice = over_slice > over.mask_threshold
            over_slice = image.colorize(over_slice, over.cmap, over.clim)

            if over.mask_image:
                mask_slice = slicer.sample(over.mask_image, interp_order)
                over_slice = image.mask(over_slice, mask_slice)
            if over.alpha_image:
                alpha_slice = slicer.sample(over.alpha_image, interp_order)
                alpha_slice = image.scale_clip(alpha_slice, over.alpha_lims)
                base_slice = image.blend(base_slice, over_slice, alpha_slice)
            else:
                alpha_slice = ones_like(over_slice)
                base_slice = image.blend(base_slice, over_slice, alpha_slice)
    if base.mask_image:
        mask_slice = slicer.sample(base.mask_image, interp_order)
        base_slice = image.mask(base_slice, mask_slice)
    return base_slice
