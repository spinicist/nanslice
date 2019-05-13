#!/usr/bin/env python
"""nanscroll.py

This implements a command-line utility (installed as nanscroll) which will create
a video scrolling through one axis of an image. This is installed by PIP as ``nanscroll``.

The majority of options are the same as :py:mod:`~nanslice.nanslicer`.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from .box import Box
from .slicer import Slicer, Axis_map
from .layer import Layer, blend_layers
from .util import add_common_arguments


def main(args=None):
    """
    The main function for the utility.

    Parameters:

    - args -- The command line-arguments
    """
    parser = argparse.ArgumentParser(
        description="Makes a video scrolling through an image")
    add_common_arguments(parser)
    parser.add_argument('output', help='Output image name', type=str)
    parser.add_argument('--slices', type=int, default=-1,
                        help='Number of slices to scroll through')
    parser.add_argument('--slice_axis', type=str, default='z',
                        help='Axis to slice along (x/y/z)')
    parser.add_argument('--slice_lims', type=float, nargs=2, default=(0.01, 0.99),
                        help='Slice between these limits along the axis, default=0.1 0.9')
    parser.add_argument('--volume', type=int, default=0,
                        help='Use this volume from a timeseries')
    parser.add_argument('--figsize', type=float, nargs=2, default=(6, 6),
                        help='Figure size (width, height) in inches')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for output figure')
    parser.add_argument('--fps', type=int, default=8,
                        help='Framerate for video (default 8)')
    parser.add_argument('--bitrate', type=int,
                        default=2048, help='Encoder bit-rate')
    args = parser.parse_args()

    print('*** Loading files')
    print('Loading base image: ', args.base_image)
    layers = [Layer(args.base_image, cmap=args.base_map, clim=args.base_lims, mask=args.mask,
                    interp_order=args.interp_order, volume=args.volume), ]
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
    args.slice_axis = Axis_map[args.slice_axis]
    if args.slices == -1:
        slices = layers[0].image.shape[args.slice_axis]
    else:
        slices = args.slices
    print(slices)
    slice_pos = bbox.start[args.slice_axis] + \
        bbox.diag[args.slice_axis] * np.linspace(args.slice_lims[0], args.slice_lims[1],
                                                 slices)
    if args.orient == 'preclin':
        origin = 'upper'
    else:
        origin = 'lower'

    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.01, right=0.99, bottom=0.01,
               top=0.99, wspace=0.01, hspace=0.01)
    fig = plt.figure(facecolor='black', figsize=args.figsize, dpi=args.dpi)

    print('*** Init Frame')
    axes = plt.subplot(gs1[0], facecolor='black')
    slicer = Slicer(
        bbox, slice_pos[0], args.slice_axis, args.samples, orient=args.orient)
    sl_final = blend_layers(layers, slicer)
    image = axes.imshow(sl_final, origin=origin,
                        extent=slicer.extent, interpolation=args.interp)
    axes.axis('off')

    def update_frame(frame):
        """Draws the next frame"""
        print('Slice pos ', slice_pos[frame])
        slicer = Slicer(bbox, slice_pos[frame], args.slice_axis,
                        args.samples, orient=args.orient)
        sl_final = blend_layers(layers, slicer)
        image.set_data(sl_final)
    print('*** Animate Frame')
    ani = FuncAnimation(fig, update_frame, frames=len(slice_pos))
    print('*** Save')
    ani.save(args.output, fps=args.fps, bitrate=args.bitrate,
             savefig_kwargs={'facecolor': 'black'})


if __name__ == "__main__":
    main()
