#!/usr/bin/env python
"""roislicer.py

Command-line tool for producing "stained glass" ROI overlay figures.
Requires an integer-labelled atlas NIfTI and a CSV file that maps each
label to a scalar value (t-statistic, p-value, group mean, etc.).

Minimum call::

    roislicer T1.nii.gz atlas.nii.gz stats.csv output.png

The CSV must have at least two columns: one with integer ROI label IDs and
one with the corresponding scalar values.  By default the first column is
taken as the label column and the second as the value column.  Use
``--label_col`` and ``--value_col`` to specify column names explicitly.

All common slicing options from ``nanslicer`` are supported.
"""
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .util import add_common_arguments, Axis_map
from .colorbar import colorbar
from .box import Box
from .slicer import Slicer
from .layer import Layer, ROILayer, blend_layers


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Stained-glass ROI overlay: colour atlas regions by a CSV value')
    # Positional arguments
    parser.add_argument('base_image', help='Background structural image', type=str)
    parser.add_argument('label_image', help='Integer-labelled atlas/ROI NIfTI', type=str)
    parser.add_argument('csv_file', help='CSV file with label → value mapping', type=str)
    parser.add_argument('output', help='Output image filename', type=str)

    # CSV column selection
    parser.add_argument('--label_col', type=str, default=None,
                        help='Column name for ROI integer labels (default: first column)')
    parser.add_argument('--value_col', type=str, default=None,
                        help='Column name for scalar values (default: second column)')

    # Base image display
    parser.add_argument('--base_map', type=str, default='gist_gray',
                        help='Colormap for the background image (default: gist_gray)')
    parser.add_argument('--base_lims', type=float, nargs=2, default=None,
                        help='Window limits for the background image')
    parser.add_argument('--mask', type=str, default=None,
                        help='Mask image for the background layer')

    # ROI overlay display
    parser.add_argument('--roi_map', type=str, default='RdYlBu_r',
                        help='Colormap for the ROI overlay (default: RdYlBu_r)')
    parser.add_argument('--roi_lim', type=float, nargs=2, default=None,
                        help='Color limits for ROI values (auto-computed if omitted)')
    parser.add_argument('--roi_label', type=str, default='',
                        help='Label for the ROI colorbar')
    parser.add_argument('--roi_alpha', type=float, default=1.0,
                        help='Overall opacity of the ROI overlay (0–1, default: 1.0)')

    # Contour options
    parser.add_argument('--no_contour', action='store_true',
                        help='Disable ROI boundary contours')
    parser.add_argument('--contour_color', type=str, default='k',
                        help='Color for ROI boundary contours (default: k = black)')
    parser.add_argument('--contour_linewidth', type=float, default=0.5,
                        help='Line width of ROI boundary contours (default: 0.5)')

    # Slicing options
    parser.add_argument('--slice_rows', type=int, default=4,
                        help='Number of rows of slices (default: 4)')
    parser.add_argument('--slice_cols', type=int, default=5,
                        help='Number of columns of slices (default: 5)')
    parser.add_argument('--slice_axis', type=str, default='z',
                        help='Axis to slice along: x/y/z (default: z)')
    parser.add_argument('--slice_lims', type=float, nargs=2, default=(0.1, 0.9),
                        help='Start and end fractions along the slice axis (default: 0.1 0.9)')
    parser.add_argument('--slices', type=float, nargs='+',
                        help='Slice at explicit positions (overrides --slice_rows/cols/lims)')
    parser.add_argument('--three_axis', action='store_true',
                        help='Show one slice per axis (x, y, z) through the image centre')
    parser.add_argument('--samples', type=int, default=128,
                        help='Sampling resolution per slice (default: 128)')
    parser.add_argument('--interp', type=str, default='hanning',
                        help='Matplotlib interpolation for display (default: hanning)')
    parser.add_argument('--orient', type=str, default='clin',
                        help='Slice orientation: clin or preclin (default: clin)')
    parser.add_argument('--radiological', action='store_true',
                        help='Flip left/right (radiological convention)')

    # Figure options
    parser.add_argument('--bar_pos', type=str, default='south',
                        help='Colorbar position: north/south/east/west (default: south)')
    parser.add_argument('--no_colorbar', action='store_true',
                        help='Do not add a colorbar')
    parser.add_argument('--figsize', type=float, nargs=2, default=None,
                        help='Figure size in inches: width height')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')
    parser.add_argument('--font', type=str, default='Helvetica',
                        help='Font family (default: Helvetica)')
    parser.add_argument('--fontsize', type=int, default=8,
                        help='Font size in pt (default: 8)')
    parser.add_argument('--title', type=str, default=None,
                        help='Optional title text on the figure')
    parser.add_argument('--crop_center', type=float, nargs=3,
                        help='Centre of crop box (X Y Z) in mm')
    parser.add_argument('--crop_size', type=float, nargs=3,
                        help='Size of crop box (X Y Z) in mm')

    args = parser.parse_args()

    mpl.rc('font', family=args.font, size=args.fontsize)

    # --- Build layers -------------------------------------------------------
    print('*** Loading background image:', args.base_image)
    bg = Layer(args.base_image,
               mask=args.mask,
               crop_center=args.crop_center,
               crop_size=args.crop_size,
               cmap=args.base_map,
               clim=args.base_lims)
    if args.base_lims is None:
        print('    Background limits:', bg.clim)

    print('*** Loading ROI layer:', args.label_image, '+', args.csv_file)
    roi = ROILayer(args.label_image, args.csv_file,
                   label_col=args.label_col,
                   value_col=args.value_col,
                   cmap=args.roi_map,
                   clim=args.roi_lim,
                   label=args.roi_label,
                   contour=not args.no_contour,
                   contour_color=args.contour_color,
                   contour_linewidth=args.contour_linewidth)
    print('    ROI value limits:', roi.clim)

    # --- Compute slice positions --------------------------------------------
    bbox = bg.bbox
    slice_axis_idx = Axis_map[args.slice_axis]

    if args.three_axis:
        slice_rows, slice_cols = 1, 3
        slice_axes = [0, 1, 2]
        slice_pos = [bbox.center[0], bbox.center[1], bbox.center[2]]
        slice_total = 3
    elif args.slices:
        slice_total = len(args.slices)
        slice_rows = args.slice_rows
        slice_cols = args.slice_cols
        slice_pos = np.array(args.slices)
        slice_axes = [slice_axis_idx] * slice_total
    else:
        slice_rows = args.slice_rows
        slice_cols = args.slice_cols
        slice_total = slice_rows * slice_cols
        slice_pos = (bbox.start[slice_axis_idx] +
                     bbox.diag[slice_axis_idx] *
                     np.linspace(args.slice_lims[0], args.slice_lims[1], slice_total))
        slice_axes = [slice_axis_idx] * slice_total

    print(f'*** Slicing: {slice_total} slices in {slice_rows} rows × {slice_cols} columns')

    origin = 'upper' if args.orient == 'preclin' else 'lower'

    # --- Layout -------------------------------------------------------------
    if not args.figsize:
        args.figsize = (3 * slice_cols, 3 * slice_rows)
    figure = plt.figure(facecolor='black', figsize=args.figsize)
    gs1 = gridspec.GridSpec(slice_rows, slice_cols)

    # --- Draw slices --------------------------------------------------------
    for s in range(slice_total):
        row, col = divmod(s, slice_cols)
        ax = plt.subplot(gs1[row, col], facecolor='black')
        slcr = Slicer(bbox, slice_pos[s], slice_axes[s],
                      args.samples, orient=args.orient)
        composite = blend_layers([bg, roi], slcr)
        ax.imshow(composite, origin=origin, extent=slcr.extent,
                  interpolation=args.interp)
        if args.radiological:
            ax.invert_xaxis()
        roi.plot_contours(slcr, ax)
        ax.axis('off')

    # --- Colorbar -----------------------------------------------------------
    if args.roi_label and not args.no_colorbar:
        print('*** Adding colorbar')
        if args.bar_pos.lower() == 'south':
            cbar_bottom = 0.3 * (args.fontsize / 12) / args.figsize[1]
            cbar_top = cbar_bottom + 0.1 / args.figsize[1]
            gs1.update(left=0.01, right=0.99, bottom=cbar_top + 0.001,
                       top=0.99, wspace=0.01, hspace=0.01)
            gs2 = gridspec.GridSpec(1, 1)
            gs2.update(left=0.1, right=0.9, bottom=cbar_bottom,
                       top=cbar_top, wspace=0.1, hspace=0.1)
            c_orient = 'h'
        elif args.bar_pos.lower() == 'north':
            cbarh = 0.15 * (args.fontsize / 12) / args.figsize[1]
            gs1.update(left=0.01, right=0.99, bottom=0.01,
                       top=0.99 - cbarh, wspace=0.01, hspace=0.01)
            gs2 = gridspec.GridSpec(1, 1)
            gs2.update(left=0.07, right=0.93, bottom=0.99 - cbarh,
                       top=0.99, wspace=0.01, hspace=0.01)
            c_orient = 'h'
        elif args.bar_pos.lower() == 'west':
            cbarw = 0.275 * (args.fontsize / 12) / args.figsize[0]
            gs1.update(left=0.01 + cbarw, right=0.99, bottom=0.01,
                       top=0.99, wspace=0.01, hspace=0.01)
            gs2 = gridspec.GridSpec(1, 1)
            gs2.update(left=0.01, right=cbarw, bottom=0.08,
                       top=0.92, wspace=0.01, hspace=0.01)
            c_orient = 'v'
        elif args.bar_pos.lower() == 'east':
            cbarw = 0.35 * (args.fontsize / 12) / args.figsize[0]
            gs1.update(left=0.01, right=1 - cbarw, bottom=0.01,
                       top=0.99, wspace=0.01, hspace=0.01)
            gs2 = gridspec.GridSpec(1, 1)
            gs2.update(left=1 - cbarw + 0.001, right=1 - cbarw / 1.5,
                       bottom=0.08, top=0.92, wspace=0.01, hspace=0.01)
            c_orient = 'v'
        else:
            raise ValueError('Unsupported bar position: ' + args.bar_pos)

        c_axes = plt.subplot(gs2[0], facecolor='black')
        colorbar(c_axes, roi.cmap, roi.clim, args.roi_label, orient=c_orient)
    else:
        gs1.update(left=0.01, right=0.99, bottom=0.01,
                   top=0.99, wspace=0.01, hspace=0.01)

    if args.title:
        figure.axes[-1].text(0.01, 0.99, args.title, color='w',
                             size=args.fontsize, verticalalignment='top',
                             transform=figure.transFigure)

    print('*** Saving:', args.output, 'at', args.dpi, 'DPI')
    figure.savefig(args.output, facecolor=figure.get_facecolor(),
                   edgecolor='none', dpi=args.dpi)
    plt.close(figure)


if __name__ == '__main__':
    main()
