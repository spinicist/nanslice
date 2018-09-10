#!/usr/bin/env python
"""slice_func.py

Functions for manipulating 'slices'/images (or (X, Y, 3) arrays)
"""

import numpy as np
import matplotlib as mpl
import scipy.ndimage.filters as filters

def colorize(data, cmap, clims=None):
    """
    Apply a colormap to grayscale data. Takes an (X, Y) array and returns an (X, Y, 3) array
    
    Parameters:

    - data -- The 2D scalar (X, Y) array to colorize
    - cmap -- Any valid matplotlib colormap or colormap name
    - clims -- The limits for the colormap
    """
    if clims is None:
        norm = None
    else:
        norm = mpl.colors.Normalize(vmin=clims[0], vmax=clims[1])
    cmap = mpl.cm.get_cmap(cmap)
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return smap.to_rgba(data, alpha=1, bytes=False)[:, :, 0:3]

def scale_clip(data, lims):
    """
    Scale an image to fill the range 0-1  and clip values that fall outside that range
    
    Parameters:

    - data -- The image data array
    - lims -- The limits to scale betwee
    """
    return np.clip((data - lims[0]) / (lims[1] - lims[0]), 0, 1)

def blend(img_under, img_over, img_alpha):
    """
    Blend together two images using an alpha channel image
    
    Parameters:
    
    - img_under -- The base image (underneath the overlay)
    - img_over  -- The overlay image
    - img_alpha -- Transparency/alpha value to use when blending
    """
    return img_under*(1 - img_alpha[:, :, None]) + img_over*img_alpha[:, :, None]

def mask(img, img_mask, back=np.array((0, 0, 0))):
    """
    Mask out sections of one image using another
    
    Parameters:

    - img -- The image to be masked
    - img_mask -- The mask image
    - back -- Background value
    """
    return blend(back, img, img_mask)

def blur(img, sigma=1):
    """
    Blur an image with a Gaussian kernel
    
    Parameters:

    - img -- The image to blur
    - sigma -- The FWHM of the Gaussian kernel, in voxels
    """
    return filters.gaussian_filter(img, sigma)

def checkerboard(img1, img2, square_size=16):
    """Combine two images in a checkerboard pattern, useful for checking image
       registration quality. Idea stolen from @ramaana_ on Twitter"""
    if (img1.shape != img2.shape):
        raise Exception('Image shape do not match:' + str(img1.shape) + ' vs:' + str(img2.shape))
    shape = img1.shape
    img3 = np.zeros_like(img1)
    from1 = True
    row = 0
    row_sz = square_size
    while row < shape[0]:
        col = 0
        col_sz = square_size
        while col < shape[1]:
            if from1:
                img3[row:row+row_sz, col:col+col_sz, :] = img1[row:row+row_sz, col:col+col_sz, :]
            else:
                img3[row:row+row_sz, col:col+col_sz, :] = img2[row:row+row_sz, col:col+col_sz, :]
            col = col + col_sz
            if (col + col_sz) > shape[1]:
                col_sz = shape[1] - col
            from1 = not from1
        row = row + row_sz
        if (row + row_sz) > shape[0]:
            row_sz = shape[0] - row
        from1 = not from1
    return img3