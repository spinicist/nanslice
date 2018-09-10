#!/usr/bin/env python
"""layer.py

Contains the :py:class:`~nanslice.layer.Layer` class and the :py:func:`~nanslice.layer.blend_layers`
function.
"""
from numpy import nanpercentile, ma
from nibabel import load
from . import slice_func
from .box import Box
from .util import ensure_image, check_path

class Layer:
    """
    The Layer class
    
    Each layer consists of a base MR image, with optional mask and alpha (transparency) images
    and their associated parameters (colormap, limits, scales etc.)

    Constructor parameters:

    - image -- The image contained in this layer. Can be either a string/path to an image file or an nibabel image
    - scale  -- A scaling factor to multiply all voxels in the image by
    - volume -- If reading a 4D file, specify which volume to use
    - interp_order -- Interpolation order. 1 is linear interpolation
    - cmap  -- The colormap to apply to the Layer. Any valid matplotlib colormap
    - clim  -- The limits (min, max) values to use for the colormap
    - label -- The label for this layer (used for colorbars)
    - mask           -- A mask image to use with this layer
    - mask_threshold -- Apply a threshold (lower) to the mask
    - alpha       -- An alpha (transparency) image to use with this layer
    - alpha_lims  -- Specify the limits/window for the alpha image
    - alpha_scale -- Scaling factor for the alpha image
    - alpha_label -- Label for the alpha axis on alphabars

    """

    def __init__(self, image, scale=1.0, volume=0, interp_order=1.0,
                 cmap=None, clim=None, label='',
                 mask=None, mask_threshold=0,
                 alpha=None, alpha_lims=None, alpha_scale=1.0, alpha_label=''):
        self.image = ensure_image(image)
        self.scale = scale
        self.interp_order = interp_order
        self.volume = volume
        self.label = label

        self.mask_image = ensure_image(mask)
        self.mask_threshold = mask_threshold
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
        self.alpha_scale = alpha_scale

    def get_slice(self, slicer):
        """
        Returns a colorized slice through the base image contained in the Layer

        Parameters:

        - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice this layer with
        """
        slc = slicer.sample(self.image, self.interp_order, self.scale, self.volume)
        color_slc = slice_func.colorize(slc, self.cmap, self.clim)
        if self.mask_image:
            mask_slc = slicer.sample(self.mask_image, 0) > self.mask_threshold
            color_slc = slice_func.mask(color_slc, mask_slc)
        return color_slc

    def get_alpha(self, slicer):
        """
        Returns the alpha (transparency) slice for this Layer

        Parameters:

        - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice this layer with
        """
        
        if self.alpha_image:
            alpha_slice = slicer.sample(self.alpha_image, self.interp_order, self.alpha_scale)
            alpha_slice = slice_func.scale_clip(alpha_slice, self.alpha_lims)
            return alpha_slice
        else:
            return None

    def plot(self, slicer, axes):
        """
        Plot a Layer into a Matplotlib axes using the provided Slicer
        
        Parameters:

        - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice this layer with
        - axes   -- A matplotlib axes object
        """
        slc = self.get_slice(slicer)
        cax = axes.imshow(slc, origin='lower', extent=slicer.extent, interpolation='nearest')
        axes.axis('off')
        return cax

def blend_layers(layers, slicer):
    """
    Blends together a set of overlays using their alpha information
    
    Parameters:

    - layers -- An iterable (e.g. list/tuple) of :py:class:`Layer` objects
    - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice the layers with    
    """
    slc = layers[0].get_slice(slicer)
    for next_layer in layers[1:]:
        next_slc = next_layer.get_slice()
        next_alpha = next_layer.get_alpha()
        if next_alpha:
            slc = slice_func.blend(slc, next_slc, next_alpha)
    return slc
