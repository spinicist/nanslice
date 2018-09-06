#!/usr/bin/env python
"""util.py

Copyright Tobias C Wood 2017

Utility functions for nanslice module"""
from pathlib import Path
import argparse
import numpy as np
from nibabel import load
from . import slice_func
from .slicer import axis_map, axis_indices

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
    idx0 = np.argmax(np.sum(img.get_data(), axis=(1,2)))
    idx1 = np.argmax(np.sum(img.get_data(), axis=(0,2)))
    idx2 = np.argmax(np.sum(img.get_data(), axis=(0,1)))
    phys = np.dot(img.affine, np.array([idx0, idx1, idx2,1]).T)
    return phys

def overlay_slice(sl, options, window,
                  img_base, img_mask,
                  img_color, img_color_mask,
                  img_alpha, volume=None):
    """Creates a slice through a base image, with a color overlay and specified alpha"""
    sl_base = slice_func.colorize(sl.sample(img_base, order=options.interp_order),
                             'gray', window)
    if img_color:
        sl_color = sl.sample(img_color, order=options.interp_order, volume=volume) * options.color_scale
        if img_color_mask:
            sl_color_mask = sl.sample(img_color_mask, order=options.interp_order)
            if options.color_mask_thresh:
                sl_color_mask = sl_color_mask > options.color_mask_thresh
        elif options.color_mask_thresh:
            sl_color_mask = sl_color > options.color_mask_thresh
        else:
            sl_color_mask = np.ones_like(sl_color)
        sl_color = slice_func.colorize(sl_color, options.color_map, options.color_lims)
        sl_color = slice_func.mask(sl_color, sl_color_mask)
        if img_alpha:
            sl_alpha = sl.sample(img_alpha, order=options.interp_order)
            sl_scaled_alpha = slice_func.scale_clip(sl_alpha, options.alpha_lims)
            sl_blend = slice_func.blend(sl_base, sl_color, sl_scaled_alpha)
        else:
            sl_blend = slice_func.blend(sl_base, sl_color, sl_color_mask)
    else:
        sl_blend = sl_base
    if img_mask:
        sl_final = slice_func.mask(sl_blend, sl.sample(img_mask, options.interp_order))
    else:
        sl_final = sl_blend
    return sl_final

def draw_slice(axis, sl, opts, window, img, mask,
               color_img=None, color_mask=None, alpha_img=None,
               contour_img=None, contour_levels=(0.95,), contour_colors='w'):
    sliced = overlay_slice(sl, opts, window, img, mask, color_img, color_mask, alpha_img)
    axis.imshow(sliced, origin='lower', extent=sl.extent, interpolation='none')
    axis.axis('off')
    if contour_img:
        sliced_contour = sl.sample(contour_img, order=1)
        if (sliced_contour < contour_levels[0]).any() and (sliced_contour > contour_levels[0]).any():
            axis.contour(sliced_contour, levels=contour_levels,
                        origin='lower', extent=sl.extent,
                        colors=contour_colors, linewidths=1)

def colorbar(axes, cm_name, clims, clabel,
             black_backg=True, show_ticks=True, tick_fmt='{:.1f}', orient='h'):
    """Plots a 2D colorbar (color/alpha)"""
    steps = 32
    if orient == 'h':
        ext = (clims[0], clims[1], 0, 1)
        cdata = np.tile(np.linspace(clims[0], clims[1], steps)[np.newaxis, :], [steps, 1])
    else:
        ext = (0, 1, clims[0], clims[1])
        cdata = np.tile(np.linspace(clims[0], clims[1], steps)[:, np.newaxis], [1, steps])
    color = slice_func.colorize(cdata, cm_name, clims)
    axes.imshow(color, origin='lower', interpolation='hanning', extent=ext, aspect='auto')
    if black_backg:
        forecolor = 'w'
    else:
        forecolor = 'k'
    if show_ticks:
        ticks = (clims[0], np.sum(clims)/2, clims[1])
        labels = (tick_fmt.format(clims[0]), clabel, tick_fmt.format(clims[1]))
        if orient == 'h':
            axes.set_xticks(ticks)
            axes.set_xticklabels(labels, color=forecolor)
            axes.set_yticks(())
        else:
            axes.set_yticks(ticks)
            axes.set_yticklabels(labels, color=forecolor, rotation='vertical', va='center')
            axes.set_xticks(())
    else:
        if orient == 'h':
            axes.set_xticks((np.sum(clims)/2,))
            axes.set_xticklabels((clabel,), color=forecolor)
        else:
            axes.set_yticks((np.sum(clims)/2,))
            axes.set_yticklabels((clabel,), color=forecolor)
    axes.tick_params(axis='both', which='both', length=0)
    axes.spines['top'].set_color(forecolor)
    axes.spines['bottom'].set_color(forecolor)
    axes.spines['left'].set_color(forecolor)
    axes.spines['right'].set_color(forecolor)
    axes.yaxis.label.set_color(forecolor)
    axes.xaxis.label.set_color(forecolor)
    axes.axis('on')

def alphabar(axes, cm_name, clims, clabel,
             alims, alabel, alines=None, alines_colors=('k',), alines_styles=('solid',),
             cprecision=1, aprecision=0,
             black_backg=True, orient='h'):
    """Plots a 2D colorbar (color/alpha)"""
    steps = 32
    if orient == 'h':
        ext = (alims[0], alims[1], clims[0], clims[1])
        cdata = np.tile(np.linspace(clims[0], clims[1], steps)[np.newaxis, :], [steps, 1])
        alpha = np.tile(np.linspace(0, 1, steps)[:, np.newaxis], [1, steps])
    else:
        ext = (alims[0], alims[1], clims[0], clims[1])
        cdata = np.tile(np.linspace(clims[0], clims[1], steps)[:, np.newaxis], [1, steps])
        alpha = np.tile(np.linspace(0, 1, steps)[np.newaxis, :], [steps, 1])
    color = slice_func.colorize(cdata, cm_name, clims)
    
    backg = np.ones((steps, steps, 3))
    acmap = slice_func.blend(backg, color, alpha)
    axes.imshow(acmap, origin='lower', interpolation='hanning', extent=ext, aspect='auto')

    cticks = (clims[0], np.sum(clims)/2, clims[1])
    cfmt = '{:.'+str(cprecision)+'f}'
    clabels = (cfmt.format(clims[0]), clabel, cfmt.format(clims[1]))
    aticks = (alims[0], np.sum(alims)/2, alims[1])
    afmt = '{:.'+str(aprecision)+'f}'
    alabels = (afmt.format(alims[0]), alabel, afmt.format(alims[1]))

    if orient == 'h':
        axes.set_xticks(cticks)
        axes.set_xticklabels(clabels)
        axes.set_yticks(aticks)
        axes.set_yticklabels(alabels, rotation='vertical')
    else:
        axes.set_xticks(aticks)
        axes.set_xticklabels(alabels)
        axes.set_yticks(cticks)
        axes.set_yticklabels(clabels, rotation='vertical', va='center')
    
    if alines:
        for ay, ac, astyle in zip(alines, alines_colors, alines_styles):
            if orient == 'h':
                axes.axhline(y=ay, linewidth=1.5, linestyle=astyle, color=ac)
            else:
                axes.axvline(y=ay, linewidth=1.5, linestyle=astyle, color=ac)
    
    if black_backg:
        axes.spines['bottom'].set_color('w')
        axes.spines['top'].set_color('w')
        axes.spines['right'].set_color('w')
        axes.spines['left'].set_color('w')
        axes.tick_params(axis='x', colors='w')
        axes.tick_params(axis='y', colors='w')
        axes.yaxis.label.set_color('w')
        axes.xaxis.label.set_color('w')
    else:
        axes.spines['bottom'].set_color('k')
        axes.spines['top'].set_color('k')
        axes.spines['right'].set_color('k')
        axes.spines['left'].set_color('k')
        axes.tick_params(axis='x', colors='k')
        axes.tick_params(axis='y', colors='k')
        axes.yaxis.label.set_color('k')
        axes.xaxis.label.set_color('k')
        axes.axis('on')

def add_common_arguments(parser):
    """Defines a set of common arguments that are shared between nanviewer and nanslicer"""
    parser.add_argument('base_image', help='Base (structural image)', type=str)
    parser.add_argument('--mask', type=str,
                        help='Mask image')
    parser.add_argument('--base_map', type=str, default='gist_gray',
                        help='Base image colormap to use from Matplotlib, default = gist_gray')
    parser.add_argument('--base_lims', type=float, nargs=2, default=None,
                        help='Specify base image window')
    parser.add_argument('--base_scale', type=float, default=1.0,
                        help='Scaling for base image before mapping, default=1.0')
    parser.add_argument('--base_label', type=str, default='',
                        help='Label for base color axis')

    parser.add_argument('--overlay', type=str,
                        help='Add color overlay')
    parser.add_argument('--overlay_lims', type=float, nargs=2, default=(-1, 1),
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
    parser.add_argument('--overlay_alpha_lims', type=float, nargs=2, default=(0.5, 1.0),
                        help='Overlay Alpha/transparency window, default=0.5 1.0')
    parser.add_argument('--overlay_alpha_label', type=str, default='1-p',
                        help='Label for overlay alpha/transparency axis')
    parser.add_argument('--overlay_contour_image', type=str,
                        help='Image to define contour (if none, use alpha image)')
    parser.add_argument('--overlay_contour', type=float, action='append',
                        help='Add an alpha image contour (can be multiple)')
    parser.add_argument('--overlay_contour_color', type=str, action='append',
                        help='Choose contour colour')
    parser.add_argument('--overlay_contour_style', type=str, action='append',
                        help='Choose contour line-style')

    parser.add_argument('--samples', type=int, default=128,
                        help='Number of samples for slicing, default=128')
    parser.add_argument('--interp', type=str, default='hanning',
                        help='Display interpolation mode, default=hanning')
    parser.add_argument('--interp_order', type=int, default=1,
                        help='Data interpolation order, default=1')
    parser.add_argument('--orient', type=str, default='clin',
                        help='Clinical (clin) or Pre-clinical (preclin) orientation')
    return parser
