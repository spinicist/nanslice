"""
nanslicer.py

A command-line tool for producing figures with slices through MR images. This is installed
by PIP as ``nanslicer``. Supports overlays, dual-coded overlays and colorbars. Dual-coding is
described here https://www.cell.com/neuron/fulltext/S0896-6273(12)00428-X.

The minimum command-line call is:

``nanslicer image.nii.gz output.png``

To add a dual-coded overlay, call:

``nanslicer structural.nii.gz output.png --overlay beta.nii.gz --overlay_alpha pval.nii.gz``

There are a lot of command-line options to control the colormaps and scaling. Type
``nanviewer --help`` to see a full list. The number of slices can be controlled with
the ``--slice_rows`` and ``--slice_cols`` arguments, or you can choose ``--three_axis``.
The ``--slice_axis`` and ``--slice_lims`` arguments specify the axis along which to slice,
and where to start and stop along it (expressed as fractions), for example:

``nanslicer structural.nii.gz --slice_axis x --slice_lims 0.25 0.75``

If you have timeseries data as the base image, you can plot the same slice through
each volume with ``--timeseries``, or you can choose the volume in the timeseries to use
with ``--volume N`` (the default is the first).

Controlling image quality is slightly complicated because there are two interpolation
steps. First we have to sample the 3D volumes to produce 2D slices to arbitrary
precision. Then, ``matplotlib`` has to sample those slices to plot them to the canvas.
The first step is controlled by ``--samples N``, which controls the number of points to sample in
each direction of the slice, and ``--interp_order N``, which controls the quality
of the interpolation. The defaults are 128 and 1 (linear interpolation). Increase
them to increase the quality. The ``matplotlib`` step is controlled by ``--interp METHOD``,
and can be any valid ``matplotlib` interpolation method. The default is ``hanning``,
for increased speed this can be changed to ``linear`` or ``none``. From experience,
it is the quality of the ``matplotlib`` step which is the dominant factor in figure
quality, hence the defaults of fairly fast sampling in the slicing step but using
Hanning sampling in the ``matplotlib`` step.
"""
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .util import add_common_arguments
from .colorbar import colorbar, alphabar
from .box import Box
from .slicer import Slicer, Axis_map
from .slice_func import scale_clip
from .layer import Layer, blend_layers


def main(args=None):
    """
    The main function that is called from the command line.

    Parameters:

    - args -- The command-line arguments. See module docstring or command-line help for a full list
    """
    parser = argparse.ArgumentParser(
        description='Takes aesthetically pleasing slices through MR images')
    add_common_arguments(parser)
    parser.add_argument('output', help='Output image name', type=str)
    parser.add_argument('--slice_rows', type=int, default=4,
                        help='Number of rows of slices')
    parser.add_argument('--slice_cols', type=int, default=5,
                        help='Number of columns of slices')
    parser.add_argument('--slice_axis', type=str, default='z',
                        help='Axis to slice along (x/y/z)')
    parser.add_argument(
        '--three_axis', help='Make a 3 axis (x,y,z) plot', action='store_true')
    parser.add_argument('--timeseries', action='store_true',
                        help='Plot the same slice through each volume in a time-series')
    parser.add_argument('--volume', type=int, default=0,
                        help='Plot one volume from a timeseries')
    parser.add_argument('--slice_lims', type=float, nargs=2, default=(0.1, 0.9),
                        help='Slice between these limits along the axis, default=0.1 0.9')
    parser.add_argument('--bar_pos', type=str, default='bottom',
                        help='Position of color-bar (bottom / right)')
    parser.add_argument('--figsize', type=float, nargs=2,
                        default=None, help='Figure size (width, height) in inches')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for output figure')
    parser.add_argument('--transpose', action='store_true',
                        help='Swap rows and columns')
    parser.add_argument('--font', type=str, default='Helvetica',
                        help='Font name, default Helvetica')
    parser.add_argument('--fontsize', type=int, default=8,
                        help='Font size, default 8')
    parser.add_argument('--title', type=str, default=None, help='Add a title')
    args = parser.parse_args()

    mpl.rc('font', family=args.font, size=args.fontsize)

    print('*** Loading base image: ', args.base_image)
    layers = [Layer(args.base_image, mask=args.mask,
                    cmap=args.base_map, clim=args.base_lims, climp=args.base_lims_p, scale=args.base_scale,
                    interp_order=args.interp_order, volume=args.volume), ]
    if args.base_lims is None:
        print('*** Base limits:', layers[0].clim)

    if args.overlay:
        layers.append(Layer(args.overlay, scale=args.overlay_scale,
                            cmap=args.overlay_map, clim=args.overlay_lim,
                            mask=args.overlay_mask, mask_threshold=args.overlay_mask_thresh,
                            alpha=args.overlay_alpha, alpha_scale=args.overlay_alpha_scale, alpha_lim=args.overlay_alpha_lim,
                            interp_order=args.interp_order))

    bbox = layers[0].bbox
    args.slice_axis = Axis_map[args.slice_axis]
    if args.three_axis:
        args.slice_rows = 1
        args.slice_cols = 3
        args.slice_axis = ['x', 'y', 'z']
        slice_total = 3
        slice_pos = (bbox.center[0], bbox.center[1], bbox.center[2])
    elif args.timeseries:
        slice_pos = bbox.center[args.slice_axis]
        slice_total = layers[0].image.shape[3]
    else:
        slice_total = args.slice_rows*args.slice_cols
        slice_pos = bbox.start[args.slice_axis] + bbox.diag[args.slice_axis] * \
            np.linspace(args.slice_lims[0], args.slice_lims[1], slice_total)
        args.slice_axis = [args.slice_axis] * slice_total
    print(slice_total, ' slices in ', args.slice_rows,
          ' rows and ', args.slice_cols, ' columns')

    if args.orient == 'preclin':
        origin = 'upper'
    else:
        origin = 'lower'

    gs1 = gridspec.GridSpec(args.slice_rows, args.slice_cols)
    if not args.figsize:
        args.figsize = (3*args.slices_cols, 2*args.slice_rows + 1)
    f = plt.figure(facecolor='black', figsize=args.figsize)

    print('*** Slicing')
    for s in range(0, slice_total):
        if args.transpose:
            col, row = divmod(s, args.slice_rows)
        else:
            row, col = divmod(s, args.slice_cols)
        ax = plt.subplot(gs1[row, col], facecolor='black')
        if args.timeseries:
            layers[0].volume = s
            sp = slice_pos
            axis = args.slice_axis
        else:
            sp = slice_pos[s]
            axis = args.slice_axis[s]

        slcr = Slicer(bbox, sp, axis, args.samples, orient=args.orient)
        sl_final = blend_layers(layers, slcr)
        ax.imshow(sl_final, origin=origin, extent=slcr.extent,
                  interpolation=args.interp)
        ax.axis('off')
        if args.contour:
            sl_contour = layers[1].get_alpha(slcr)
            contour_levels = scale_clip(
                np.array(args.contour), args.overlay_alpha_lim)

            # Contour levels must be within the range of overlay alpha values.
            # Ignore contour levels that are not within this range to prevent
            # spurious contour lines from being drawn.
            valid_levels = (np.min(sl_contour) < contour_levels) & (
                contour_levels < np.max(sl_contour))
            if any(valid_levels):
                ax.contour(sl_contour, levels=contour_levels[valid_levels], origin=origin, extent=slcr.extent,
                           colors=args.contour_color, linestyles=args.contour_style, linewidths=1)

    if args.base_label or args.overlay_label:
        print('*** Adding colorbar')
        # Need about 1/4 inch space
        if args.bar_pos == 'bottom':
            cbarh = (1/4) / args.figsize[1]
            gs1.update(left=0.01, right=0.99, bottom=cbarh+0.001,
                       top=0.99, wspace=0.01, hspace=0.01)
            gs2 = gridspec.GridSpec(1, 1)
            gs2.update(left=0.08, right=0.92, bottom=cbarh/2,
                       top=cbarh, wspace=0.1, hspace=0.1)
            orient = 'h'
        else:
            cbarw = (1/4) / args.figsize[0]
            gs1.update(left=0.01, right=1 - cbarw, bottom=0.01,
                       top=0.99, wspace=0.01, hspace=0.01)
            gs2 = gridspec.GridSpec(1, 1)
            gs2.update(left=1 - cbarw + 0.001, right=1 - cbarw/1.5, bottom=0.08,
                       top=0.92, wspace=0.01, hspace=0.01)
            orient = 'v'
        axes = plt.subplot(gs2[0], facecolor='black')
        if args.overlay_alpha:
            alphabar(axes, args.overlay_map, args.overlay_lim, args.overlay_label,
                     args.overlay_alpha_lim, args.overlay_alpha_label, orient=orient)
        else:
            if args.base_map:
                colorbar(axes, layers[0].cmap, layers[0].clim,
                         args.base_label, orient=orient)
            else:
                colorbar(axes, layers[1].cmap, layers[1].clim,
                         args.overlay_label, orient=orient)
    else:
        gs1.update(left=0.01, right=0.99, bottom=0.01,
                   top=0.99, wspace=0.01, hspace=0.01)

    if args.title:
        f.axes[-1].text(0.01, 0.99, args.title, color='w', size=args.fontsize, verticalalignment='top',
                        transform=f.transFigure)
    print('*** Saving')
    print('Writing file: ', args.output, 'at', args.dpi, ' DPI')
    f.savefig(args.output, facecolor=f.get_facecolor(),
              edgecolor='none', dpi=args.dpi)
    plt.close(f)


if __name__ == "__main__":
    main()
