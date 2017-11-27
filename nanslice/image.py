#!/usr/bin/env python
"""image.py

Functions for manipulating images (in this (X, Y, 3) arrays)"""

import numpy as np
import scipy.ndimage.filters as filters
import matplotlib as mpl

def colorize(data, cm_name, clims=None):
    """Apply a colormap to grayscale data. Takes an (X, Y) array and returns an (X, Y, 3) array"""
    if clims is None:
        norm = None
    else:
        norm = mpl.colors.Normalize(vmin=clims[0], vmax=clims[1])
    cmap = mpl.cm.get_cmap(cm_name)
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return smap.to_rgba(data, alpha=1, bytes=False)[:, :, 0:3]

def scale_clip(data, lims):
    """Scale an image to fill the range 0-1  and clip values that fall outside that range"""
    return np.clip((data - lims[0]) / (lims[1] - lims[0]), 0, 1)

def blend(img_under, img_over, img_alpha):
    """Blend together two images using an alpha channel image"""
    return img_under*(1 - img_alpha[:, :, None]) + img_over*img_alpha[:, :, None]

def mask(img, img_mask, back=np.array((0, 0, 0))):
    """Mask out sections of one image using another"""
    return blend(back, img, img_mask)

def blur(img, sigma=1):
    """Blur an image with a Gaussian kernel"""
    return filters.gaussian_filter(img, sigma)
