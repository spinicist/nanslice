#!/usr/bin/env python
"""layer.py

Contains the :py:class:`~nanslice.layer.Layer` class and the :py:func:`~nanslice.layer.blend_layers`
function.
"""
import scipy.ndimage.interpolation as ndinterp
import h5py
from numpy import isfinite, nanpercentile, ma, ones_like, array, iscomplexobj, abs, angle, mat, dot, eye
from nibabel import load
from . import slice_func
from .box import Box
from .util import ensure_image, check_path


def get_component(data, component):
    data[~isfinite(data)] = 0
    if iscomplexobj(data):
        if component is None:
            data = data.real
        elif component == 'real':
            data = data.real
        elif component == 'imag':
            data = data.imag
        elif component == 'mag':
            data = abs(data)
        elif component == 'phase':
            data = angle(data)
        else:
            raise('Unknown component type ' + component)
    return data


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
    - crop_center -- Center of box to crop to
    - crop_size   -- Size of crop box
    - alpha       -- An alpha (transparency) image to use with this layer
    - alpha_lim  -- Specify the limits/window for the alpha image
    - alpha_scale -- Scaling factor for the alpha image
    - alpha_label -- Label for the alpha axis on alphabars
    - background -- Background color for masking, either 'black' (default) or 'white'

    """

    def __init__(self, image, scale=1.0, volume=0, interp_order=1,
                 cmap=None, clim=None, climp=None, label='', component=None,
                 mask=None, mask_threshold=0, crop_center=None, crop_size=None,
                 alpha=None, alpha_lim=None, alpha_scale=1.0, alpha_label='',
                 background='black'):
        self.scale = scale
        self.interp_order = interp_order
        self.volume = volume
        self.label = label

        image = ensure_image(image)
        self.affine = image.affine
        self.img_data = get_component(image.get_data(), component)
        self.matrix = self.img_data.shape

        self.mask_image = ensure_image(mask)
        self.mask_threshold = mask_threshold
        if crop_center and crop_size:
            self.bbox = Box(center=crop_center, size=crop_size)
        elif self.mask_image:
            self.bbox = Box.fromMask(self.mask_image)
        else:
            self.bbox = Box.fromImage(self.matrix, self.affine)

        if clim is not None:
            self.clim = clim
        else:
            if len(self.matrix) == 4:
                limdata = self.img_data[:, :, :, self.volume].squeeze()
            else:
                limdata = self.img_data
            if self.mask_image:
                limdata = ma.masked_where(
                    self.mask_image.get_data() == 0, limdata).compressed()
            if climp is None:
                climp = (2, 98)
            self.clim = nanpercentile(limdata, climp)

        if cmap:
            self.cmap = cmap
        elif self.clim[0] < 0 and self.clim[1] > 0:
            self.cmap = 'twoway'
        else:
            self.cmap = 'gist_gray'

        if check_path(alpha):
            self.alpha_image = load(str(alpha))
            if alpha_lim is None:
                self.alpha_lim = nanpercentile(
                    abs(self.alpha_image.get_data()), (2, 98))
            else:
                self.alpha_lim = alpha_lim

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

    def get_value(self, pos):
        """"
        Returns the value of the image at the given position

        Parameters:

        - pos -- The position to sample the image value at
        """
        pos = mat(pos).T
        scale = mat(self.affine[0:3, 0:3]).I
        offset = dot(-scale, self.affine[0:3, 3]).T
        vox = dot(scale, pos) + offset
        return ndinterp.map_coordinates(self.img_data, vox, order=1)[0]

    def get_slice(self, slicer):
        """
        Returns a slice through the base image

        Parameters:

        - slicer -- The :py:class:`~nanslice.slicer.Slicer` object to slice this layer with
        """
        vals = slicer.sample(self.img_data, self.affine,
                             self.interp_order, self.scale, self.volume)
        return vals

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
                self.img_data, self.affine, self.interp_order, self.scale, self.volume) > self.mask_threshold
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
            alpha_slice = abs(slicer.sample(
                self.alpha_image.get_data(), self.alpha_image.affine, self.interp_order, self.alpha_scale, self.volume))
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


class H5Layer(Layer):
    """
    Read a Layer from an HDF5 dataset. Has a different constructor to normal
    layers.

    - path -- Path to .h5 file
    - ds   -- Dataset path within the .h5 file
    - slices -- A list of (dim, index) pairs to slice through
    - scale  -- A scaling factor to multiply all voxels in the image by
    - volume -- If reading a 4D file, specify which volume to use
    - interp_order -- Interpolation order. 1 is linear interpolation
    - cmap  -- The colormap to apply to the Layer. Any valid matplotlib colormap
    - clim  -- The limits (min, max) values to use for the colormap
    - label -- The label for this layer (used for colorbars)
    - mask           -- A mask image to use with this layer
    - mask_threshold -- Apply a threshold (lower) to the mask
    - crop_center -- Center of box to crop to
    - crop_size   -- Size of crop box
    - alpha       -- An alpha (transparency) image to use with this layer
    - alpha_lim  -- Specify the limits/window for the alpha image
    - alpha_scale -- Scaling factor for the alpha image
    - alpha_label -- Label for the alpha axis on alphabars
    - background -- Background color for masking, either 'black' (default) or 'white'

    """

    def __init__(self, path, ds, slices=None, scale=1.0, volume=0, interp_order=1,
                 cmap=None, clim=None, climp=None, label='', component=None,
                 mask=None, mask_threshold=0, crop_center=None, crop_size=None,
                 alpha=None, alpha_lim=None, alpha_scale=1.0, alpha_label='',
                 background='black'):
        self.scale = scale
        self.interp_order = interp_order
        self.volume = volume
        self.label = label

        self.affine = eye(4)
        h5file = h5py.File(path, 'r')
        h5ds = h5file[ds]

        sl = [slice(None), ]*h5ds.ndim
        if slices:
            for dimSlice in slices:
                sl[dimSlice[0]] = dimSlice[1]
            self.img_data = h5ds[tuple(sl)]
        else:
            self.img_data = array(h5ds)
        self.img_data = get_component(self.img_data, component)
        self.matrix = self.img_data.shape

        self.mask_image = ensure_image(mask)
        self.mask_threshold = mask_threshold
        if crop_center and crop_size:
            self.bbox = Box(center=crop_center, size=crop_size)
        elif self.mask_image:
            self.bbox = Box.fromMask(self.mask_image)
        else:
            self.bbox = Box.fromImage(self.matrix, self.affine)

        if clim is not None:
            self.clim = clim
        else:
            if len(self.matrix) == 4:
                limdata = self.img_data[:, :, :, self.volume].squeeze()
            else:
                limdata = self.img_data
            if self.mask_image:
                limdata = ma.masked_where(
                    self.mask_image.get_data() == 0, limdata).compressed()
            if climp is None:
                climp = (2, 98)
            self.clim = nanpercentile(limdata, climp)

        if cmap:
            self.cmap = cmap
        elif self.clim[0] < 0 and self.clim[1] > 0:
            self.cmap = 'twoway'
        else:
            self.cmap = 'gist_gray'

        if check_path(alpha):
            self.alpha_image = load(str(alpha))
            if alpha_lim is None:
                self.alpha_lim = nanpercentile(
                    abs(self.alpha_image.get_data()), (2, 98))
            else:
                self.alpha_lim = alpha_lim

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

        h5file.close()
