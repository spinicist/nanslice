#!/usr/bin/env python
"""util.py

Utility functions for nanslice module
"""
from pathlib import Path
import numpy as np
from nibabel import load


def check_path(maybe_path):
    """Helper function to check if an object is path-like"""
    if isinstance(maybe_path, Path) or isinstance(maybe_path, str):
        return True
    else:
        return False


def ensure_image(maybe_path):
    """Helper function that lets either images or paths be passed around"""
    if check_path(maybe_path):
        return load(str(maybe_path))
    else:
        return maybe_path


def center_of_mass(img):
    """Calculates the center of mass of the image"""
    idx0 = np.argmax(np.sum(img.get_data(), axis=(1, 2)))
    idx1 = np.argmax(np.sum(img.get_data(), axis=(0, 2)))
    idx2 = np.argmax(np.sum(img.get_data(), axis=(0, 1)))
    phys = np.dot(img.affine, np.array([idx0, idx1, idx2, 1]).T)
    return phys


def add_common_arguments(parser):
    """Defines a set of common arguments that are shared between nanviewer and nanslicer"""
    parser.add_argument('base_image', help='Base (structural image)', type=str)
    parser.add_argument('--mask', type=str,
                        help='Mask image')
    parser.add_argument('--base_map', type=str, default='gist_gray',
                        help='Base image colormap to use from Matplotlib, default = gist_gray')
    parser.add_argument('--base_lims', type=float, nargs=2, default=None,
                        help='Specify base image window')
    parser.add_argument('--base_lims_p', type=float, nargs=2, default=None,
                        help='Specify base image window in %')
    parser.add_argument('--base_scale', type=float, default=1.0,
                        help='Scaling for base image before mapping, default=1.0')
    parser.add_argument('--base_label', type=str, default='',
                        help='Label for base color axis')

    parser.add_argument('--overlay', type=str,
                        help='Add color overlay')
    parser.add_argument('--overlay_map', type=str, default='RdYlBu_r',
                        help='Overlay colormap, default = RdYlBu_r')
    parser.add_argument('--overlay_lim', type=float, nargs=2, default=(-1, 1),
                        help='Overlay window, default=-1 1')
    parser.add_argument('--overlay_mask', type=str,
                        help='Mask color image')
    parser.add_argument('--overlay_mask_thresh', type=float,
                        help='Overlay mask threshold')
    parser.add_argument('--overlay_scale', type=float, default=1.0,
                        help='Scaling for overlay image before mapping, default=1.0')
    parser.add_argument('--overlay_label', type=str, default='',
                        help='Label for overlay color axis')
    parser.add_argument('--overlay_alpha', type=str,
                        help='Image for transparency-coding of overlay')
    parser.add_argument('--overlay_alpha_lim', type=float, nargs=2, default=(0.5, 1.0),
                        help='Overlay Alpha/transparency window, default=0.5 1.0')
    parser.add_argument('--overlay_alpha_scale', type=float, default=1.0,
                        help='Scaling factor for the alpha image')
    parser.add_argument('--overlay_alpha_label', type=str, default='1-p',
                        help='Label for overlay alpha/transparency axis')
    parser.add_argument('--contour', type=float, action='append',
                        help='Add alpha image contours (can be multiple)')
    parser.add_argument('--contour_color', type=str, action='append', default='k',
                        help='Choose contour colors')
    parser.add_argument('--contour_style', type=str, action='append', default='-',
                        help='Choose contor line-styles')

    parser.add_argument('--samples', type=int, default=128,
                        help='Number of samples for slicing, default=128')
    parser.add_argument('--interp', type=str, default='hanning',
                        help='Display interpolation mode, default=hanning')
    parser.add_argument('--interp_order', type=int, default=1,
                        help='Data interpolation order, default=1')
    parser.add_argument('--orient', type=str, default='clin',
                        help='Clinical (clin) or Pre-clinical (preclin) orientation')
    return parser
