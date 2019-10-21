#!/usr/bin/env python
"""layer.py

Contains the :py:class:`~nanslice.layer.Layer` class and the :py:func:`~nanslice.layer.blend_layers`
function.
"""
from numpy import isfinite, nanpercentile, ma, ones_like, array
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
    - alpha_lim  -- Specify the limits/window for the alpha image
    - alpha_scale -- Scaling factor for the alpha image
    - alpha_label -- Label for the alpha axis on alphabars
    - background -- Background color for masking, either 'black' (default) or 'white'

    """

    def __init__(self, image, scale=1.0, volume=0, interp_order=1,
                 cmap=None, clim=None, climp=None, label='',
                 mask=None, mask_threshold=0,
                 alpha=None, alpha_lim=None, alpha_scale=1.0, alpha_label='',
                 background='black'):
        self.image = ensure_image(image)
        self.scale = scale
        self.interp_order = interp_order
        self.volume = volume
        self.label = label

        self.img_data = self.image.get_data()
        self.img_data[~isfinite(self.img_data)] = 0

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
            if len(self.img_data.shape) == 4:
                limdata = self.img_data[:, :, :, self.volume].squeeze()
            else:
                limdata = self.img_data
            if self.mask_image:
                limdata = ma.masked_where(
                    self.mask_image.get_data() == 0, limdata).compressed()
            if climp is None:
                climp = (2, 98)
            self.clim = nanpercentile(limdata, climp)

        if check_path(alpha):
            self.alpha_image = load(str(alpha))
            if alpha_lim:
                self.alpha_lim = alpha_lim
            else:
                self.alpha_lim = nanpercentile(
                    self.alpha_image.get_data(), (2, 98))
        elif alpha:
            self.alpha_image = ones_like(self.image) * alpha
        else:
            self.alpha_image = None
        self.alpha_label = alpha_label
        self.alpha_scale = alpha_scale

        if background == 'white':
            self._back = array([1])
        else:
            self._back = array([0])

    def get_slice(self, slicer):
        """
        Returns a slice through the base image

        Parameters:

        - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice this layer with
        """
        return slicer.sample(self.img_data, self.image.affine, self.interp_order, self.scale, self.volume)

    def get_color(self, slicer):
        """
        Returns a colorized slice through the base image contained in the Layer

        Parameters:

        - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice this layer with
        """
        return slice_func.colorize(self.get_slice(slicer), self.cmap, self.clim)

    def get_mask(self, slicer):
        if self.mask_image:
            mask_slc = slicer.sample(self.mask_image.get_data(
            ), self.mask_image.affine, 0) > self.mask_threshold
        elif self.mask_threshold:
            mask_slc = slicer.sample(
                self.img_data, self.image.affine, self.interp_order, self.scale, self.volume) > self.mask_threshold
        else:
            return None
        return mask_slc

    def get_alpha(self, slicer):
        """
        Returns the alpha (transparency) slice for this Layer

        Parameters:

        - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice this layer with
        """

        if self.alpha_image:
            alpha_slice = slicer.sample(
                self.alpha_image.get_data(), self.alpha_image.affine, self.interp_order, self.alpha_scale)
            alpha_slice = slice_func.scale_clip(alpha_slice, self.alpha_lim)
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
        slc = slice_func.mask(self.get_color(
            slicer), self.get_mask(slicer), back=self._back)
        cax = axes.imshow(slc, origin='lower',
                          extent=slicer.extent, interpolation='nearest')
        axes.axis('off')
        return cax


def blend_layers(layers, slicer):
    """
    Blends together a set of overlays using their alpha information

    Parameters:

    - layers -- An iterable (e.g. list/tuple) of :py:class:`Layer` objects
    - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice the layers with    
    """
    slc = slice_func.mask(layers[0].get_color(
        slicer), layers[0].get_mask(slicer))
    for next_layer in layers[1:]:
        next_slc = next_layer.get_color(slicer)
        if next_layer.alpha_image:
            next_alpha = next_layer.get_alpha(slicer)
            slc = slice_func.blend(slc, next_slc, next_alpha)
        else:
            slc = slice_func.mask(next_slc, next_layer.get_mask(slicer), slc)
    return slc
