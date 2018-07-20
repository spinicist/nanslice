#!/usr/bin/env python
# Copyright (C) 2017 Tobias Wood

# This code is subject to the terms of the Mozilla Public License. A copy can be
# found in the root directory of the project.
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from .util import overlay_slice, alphabar, colorbar
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

    parser.add_argument('--slice_axis', type=str, default='z', help='Axis to slice along (x/y/z)')
    parser.add_argument('--slice_lims', type=float, nargs=2, default=(0.01, 0.99),
                        help='Slice between these limits along the axis, default=0.1 0.9')
    parser.add_argument('--figsize', type=float, nargs=2, default=(6, 6), help='Figure size (width, height) in inches')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output figure')
    args = parser.parse_args()

    print('*** Loading files')
    print('Loading base image: ', args.base_image)
    layers = [Layer(args.base_image, cmap=args.base_map, clim=args.base_lims, mask=args.mask,
                 interp_order=args.interp_order),]
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
    slice_total = 128
    args.slice_axis = [args.slice_axis] * slice_total
    slice_pos = bbox.start + bbox.diag * np.linspace(args.slice_lims[0], args.slice_lims[1], slice_total)[:, np.newaxis]
    print(slice_total, ' slices')

    if args.orient == 'preclin':
        origin = 'upper'
    else:
        origin = 'lower'

    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
    fig = plt.figure(facecolor='black', figsize=(6, 6))

    print('*** Init Frame')
    ax = plt.subplot(gs1[0], facecolor='black')
    sp = slice_pos[0, :]
    axis = args.slice_axis[0]
    sl = Slicer(bbox, sp, axis, args.samples, orient=args.orient)
    sl_final = blend_layers(layers, sl)
    im = ax.imshow(sl_final, origin=origin, extent=sl.extent, interpolation=args.interp)
    ax.axis('off')

    def update_frame(s):
        sp = slice_pos[s, :]
        axis = args.slice_axis[s]
        print('Slice pos ', sp)
        sl = Slicer(bbox, sp, axis, args.samples, orient=args.orient)
        sl_final = blend_layers(layers, sl)
        im.set_data(sl_final)
    print('*** Animate Frame')
    ani = FuncAnimation(fig, update_frame, frames=slice_total)
    print('*** Save')
    ani.save(args.output, fps=8, bitrate=2048, savefig_kwargs={'facecolor':'black'})

if __name__ == "__main__":
    main()
