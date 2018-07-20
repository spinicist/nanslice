#!/usr/bin/env python
# Copyright (C) 2017 Tobias Wood

# This code is subject to the terms of the Mozilla Public License. A copy can be
# found in the root directory of the project.
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .util import alphabar, colorbar
from .box import Box
from .slicer import Slicer
from .layer import Layer, blend_layers
def main(args=None):
    parser = argparse.ArgumentParser(description='Dual-coding viewer.')
    parser.add_argument('base_image', help='Base (structural image)', type=str)
    parser.add_argument('output', help='Output image name', type=str)
    parser.add_argument('--mask', type=str,
                        help='Mask image')
    parser.add_argument('--base_map', type=str, default='gist_gray',
                        help='Base image colormap to use from Matplotlib, default = RdYlBu_r')
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

    parser.add_argument('--slice_rows', type=int, default=4, help='Number of rows of slices')
    parser.add_argument('--slice_cols', type=int, default=5, help='Number of columns of slices')
    parser.add_argument('--slice_axis', type=str, default='z', help='Axis to slice along (x/y/z)')
    parser.add_argument('--three_axis', help='Make a 3 axis (x,y,z) plot', action='store_true')
    parser.add_argument('--timeseries', action='store_true', help='Plot the same slice through each volume in a time-series')
    parser.add_argument('--volume', type=int, default=1, help='Plot one volume from a timeseries')
    parser.add_argument('--slice_lims', type=float, nargs=2, default=(0.1, 0.9),
                        help='Slice between these limits along the axis, default=0.1 0.9')
    parser.add_argument('--bar_pos', type=str, default='bottom', help='Position of color-bar (bottom / right)')
    parser.add_argument('--figsize', type=float, nargs=2, default=None, help='Figure size (width, height) in inches')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output figure')
    args = parser.parse_args()

    print('*** Loading files')
    print('Loading base image: ', args.base_image)
    layers = [Layer(args.base_image, cmap=args.base_map, clim=args.base_lims, mask=args.mask,
                    interp_order=args.interp_order,volume=args.volume),]
    if args.base_lims is None:
        print('Base limits:', layers[0].clim)
    
    if args.overlay:
        layers.append(Layer(args.overlay, cmap=args.overlay_map, clim=args.overlay_lims,
                            mask=args.overlay_mask, mask_threshold=args.overlay_mask_thresh,
                            alpha=args.alpha, alpha_lims=args.alpha_lims,
                            interp_order=args.interp_order))

    print('*** Setup')
    if layers[0].mask_image:
        bbox = Box.fromMask(layers[0].mask_image)
    else:
        bbox = Box.fromImage(layers[0].image)
    print(bbox)
    if args.three_axis:
        args.slice_rows = 1
        args.slice_cols = 3
        args.slice_axis = ['x', 'y', 'z']
        slice_total = 3
        slice_pos = np.tile(bbox.center, (3, 1))
    elif args.timeseries:
        # slice_pos = bbox.center
        slice_pos = bbox.start + bbox.diag * 0.4
        slice_total = layers[0].image.shape[3]
    else:
        slice_total = args.slice_rows*args.slice_cols
        args.slice_axis = [args.slice_axis] * slice_total
        slice_pos = bbox.start + bbox.diag * np.linspace(args.slice_lims[0], args.slice_lims[1], slice_total)[:, np.newaxis]
    print(slice_total, ' slices in ', args.slice_rows, ' rows and ', args.slice_cols, ' columns')

    if args.orient == 'preclin':
        origin = 'upper'
    else:
        origin = 'lower'

    gs1 = gridspec.GridSpec(args.slice_rows, args.slice_cols)
    if args.figsize:
        f = plt.figure(facecolor='black', figsize=args.figsize)
    else:
        f = plt.figure(facecolor='black', figsize=(3*args.slice_cols, 3*args.slice_rows))

    print('*** Slicing')
    for s in range(0, slice_total):
        ax = plt.subplot(gs1[s], facecolor='black')
        if args.timeseries:
            layers[0].volume = s
            sp = slice_pos
            axis = args.slice_axis
        else:
            sp = slice_pos[s, :]
            axis = args.slice_axis[s]
        
        print('Slice pos ', sp)
        sl = Slicer(bbox, sp, axis, args.samples, orient=args.orient)
        sl_final = blend_layers(layers, sl)
        ax.imshow(sl_final, origin=origin, extent=sl.extent, interpolation=args.interp)
        ax.axis('off')
        # if img_contour:
        #     sl_contour = sl.sample(img_contour, order=args.interp_order)
        #     ax.contour(sl_contour, levels=args.contour, origin=origin, extent=sl.extent,
        #             colors=args.contour_color, linestyles=args.contour_style, linewidths=1)

    if args.base_label or args.overlay_label:
        print('*** Adding colorbar')
        if args.bar_pos == 'bottom':
            gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
            gs2 = gridspec.GridSpec(1, 1)
            gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.15, wspace=0.1, hspace=0.1)
            orient='h'
        else:
            gs1.update(left=0.01, right=0.95, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
            gs2 = gridspec.GridSpec(1, 1)
            gs2.update(left=0.97, right=0.99, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)
            orient='v'
        axes = plt.subplot(gs2[0], facecolor='black')
        if args.overlay_alpha:
            alphabar(axes, args.overlay_map, args.overlay_lims, args.overlay_label,
                        args.overlay_alpha_lims, args.overlay_alpha_label, orient=orient)
        else:
            if args.base_map:
                colorbar(axes, layers[0].cmap, layers[0].clim, args.base_label, orient=orient)
            else:
                colorbar(axes, layers[1].cmap, layers[1].clim, args.overlay_label, orient=orient)
    else:
        gs1.update(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
    print('*** Saving')
    print('Writing file: ', args.output, 'at', args.dpi, ' DPI')
    f.savefig(args.output, facecolor=f.get_facecolor(), edgecolor='none', dpi=args.dpi)
    plt.close(f)

if __name__ == "__main__":
    main()
