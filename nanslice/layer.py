#!/usr/bin/env python
"""layer.py

Contains the Layer class which stores settings for each layer (image to sample,
cmap, transparency, etc., and the overlay function for combining layers
"""
from numpy import nanpercentile, ma
from nibabel import load
from . import image_func
from .box import Box
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

        self.mask_image = ensure_image(mask)
        if mask_threshold:
            self.mask_threshold = mask_threshold
        else:
            self.mask_threshold = None
        if self.mask_image:
            self.bbox = Box.fromMask(self.mask_image)
        else:
            self.bbox = Box.fromImage(self.image)

        if cmap:
            self.cmap = cmap
        else:
            self.cmap = 'gist_gray'
        if clim:
            self.clim = clim
        else:
            if len(self.image.shape) == 4:
                imdata = self.image.dataobj[:,:,:,self.volume].squeeze()
            else:
                imdata = self.image.dataobj
            if self.mask_image:
                imdata = ma.masked_where(self.mask_image.get_data() > 0, imdata).compressed()
            self.clim = nanpercentile(imdata, (2, 98))

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
        slc = image_func.colorize(slicer.sample(self.image, self.interp_order, self.scale, self.volume),
                             self.cmap, self.clim)
        if self.mask_image:
            mask_slc = slicer.sample(self.mask_image, 0) > 0
            slc = image_func.mask(slc, mask_slc)
        return slc

    def get_alpha(self, slicer):
        if self.alpha_image:
            alpha_slice = slicer.sample(self.alpha_image, self.interp_order)
            alpha_slice = image_func.scale_clip(alpha_slice, self.alpha_lims)
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
    slc = layers[0].get_slice(slicer)
    for l in layers[1:]:
        next_slc = l.get_slice()
        next_alpha = l.get_alpha()
        if next_alpha:
            slc = image_func.blend(slc, next_slc, next_alpha)
    return slc
