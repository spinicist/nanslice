#!/usr/bin/env python
"""layer.py

Contains the Layer class which stores settings for each layer (image to sample,
cmap, transparency, etc., and the overlay function for combining layers
"""
from numpy import nanpercentile, ones_like
from nibabel import load
from . import image
from .util import ensure_image, check_path

class Layer:
    """The Layer class. Keeps tabs on the image, color-map & transparency value"""

    def __init__(self, image, scale=1.0, volume=0, interp_order=1.0,
                 cmap=None, clim=None, label='',
                 mask=None, mask_threshold=None,
                 alpha=None, alpha_lims=None, alpha_scale=1.0, alpha_label=''):
        self.image = ensure_image(image)
        self.scale = scale
        self.interp_order = interp_order
        self.volume = volume
        self.label = label
        if cmap:
            self.cmap = cmap
        else:
            self.cmap = 'gist_gray'
        if clim:
            self.clim = clim
        else:
            self.clim = nanpercentile(self.image.get_data(), (2, 98))
        self.mask_image = ensure_image(mask)
        if mask_threshold:
            self.mask_threshold = mask_threshold
        else:
            self.mask_threshold = None
        if self.mask_image:
            self.bbox = Box.fromMask(self.mask_image)
        else:
            self.bbox = Box.fromImage(self.image)
        if check_path(alpha):
            self.alpha_image = load(str(alpha))
            if alpha_lims:
                self.alpha_lims = alpha_lims
            else:
                self.alpha_lims = nanpercentile(self.alpha_image.get_data(), (2, 98))
            self.alpha_label = alpha_label
        elif alpha:
            self.alpha_image = None
            self.alpha = alpha
            self.alpha_label = alpha_label
        else:
            self.alpha_image = None
            self.alpha = 1.0
            self.alpha_label = alpha_label

    def get_slice(self, slicer):
        """Return the slice for this Layer"""
        slc = image.colorize(slicer.sample(self.image, self.interp_order, self.scale, self.volume),
                             self.cmap, self.clim)
        if self.mask_image:
            mask_slc = slicer.sample(self.mask_image, 0) > 0
            slc = image.mask(slc, mask_slc)
        return slc

    def get_alpha(self, slicer):
        if self.alpha_image:
            alpha_slice = slicer.sample(self.alpha_image, self.interp_order)
            alpha_slice = image.scale_clip(alpha_slice, self.alpha_lims)
            return alpha_slice
        else:
            return None

    def plot(self, slicer, axes):
        """Plot a Layer into a Matplotlib axes using the provided Slicer"""
        slc = self.get_slice(slicer)
        cax = axes.imshow(slc, origin='lower', extent=slicer.extent, interpolation='nearest')
        axes.axis('off')
        return cax


def blend_layers(layers, slicer):
    """Blends together a set of overlays"""
    base_slice = image.colorize(slicer.sample(base.image, interp_order, base.scale, base.volume) * base.scale,
                                base.cmap, base.clim)
    if overlays:
        for over in overlays:
            over_slice = slicer.sample(over.image, interp_order, over.scale, over.volume)
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
        mask_slice = slicer.sample(base.mask_image, interp_order) > 0
        base_slice = image.mask(base_slice, mask_slice)
    return base_slice
