#!/usr/bin/env python
# Copyright (C) 2017 Tobias Wood

# This code is subject to the terms of the Mozilla Public License. A copy can be
# found in the root directory of the project.
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qicommon as qi

# pylint insists anything at module level is a constant, so disable the stupidity
# pylint: disable=C0103
parser = qi.common_args()
parser.add_argument('output',help='Output image name',type=str)
parser.add_argument('--slice_rows', type=int, default=4, help='Number of rows of slices')
parser.add_argument('--slice_cols', type=int, default=5, help='Number of columns of slices')
parser.add_argument('--slice_axis', type=str, default='z', help='Axis to slice along (x/y/z)')
parser.add_argument('--slice_lims', type=float, nargs=2, default=(0.1,0.9), help='Slice between these limits along the axis, default=0.1 0.9')
parser.add_argument('--figsize', type=float, nargs=2, default=(6, 4), help='Figure size in inches')
args = parser.parse_args()

print('*** Loading files')
print('Loading base image: ', args.base_image)
img_base = nib.load(args.base_image)
img_mask = None
img_color = None
img_color_mask = None
img_alpha = None
if args.mask:
    print('Loading mask image: ', args.mask)
    img_mask = nib.load(args.mask)
if args.color:
    print('Loading color overlay image: ', args.color)
    img_color = nib.load(args.color)
    if args.color_mask:
        print('Loading color mask image: ', args.color_mask)
        img_color_mask = nib.load(args.color_mask)
    if args.alpha:
        print('Loading alpha image: ', args.alpha)
        img_alpha = nib.load(args.alpha)

print('*** Setup')
window = np.percentile(img_base.get_data(), args.window)
print('Base image window: ', window[0], ' - ', window[1])
if args.mask:
    (corner1, corner2) = qi.mask_bbox(img_mask)
else:
    (corner1, corner2) = qi.img_bbox(img_base)
print('Bounding box: ', corner1, ' -> ', corner2)
slice_total = args.slice_rows*args.slice_cols
print(slice_total, ' slices in ', args.slice_rows, ' rows and ', args.slice_cols, ' columns')
slice_pos = np.linspace(args.slice_lims[0], args.slice_lims[1], slice_total)

gs1 = gridspec.GridSpec(args.slice_rows, args.slice_cols)
f = plt.figure(facecolor='black', figsize=args.figsize)

print('*** Slicing')
for s in range(0, slice_total):
    ax = plt.subplot(gs1[s], facecolor='black')
    print('Slice pos ', slice_pos[s])
    sl = qi.Slice(corner1, corner2, args.slice_axis, slice_pos[s], args.samples, orient=args.orient)
    (sl_final, sl_alpha) = qi.overlay_slice(sl, args, window,
                                            img_base, img_mask, img_color, img_color_mask, img_alpha)
    ax.imshow(sl_final, origin='lower', extent=sl.extent, interpolation=args.interp)
    ax.axis('off')
    if args.alpha and args.contour:
        ax.contour(sl_alpha, levels=args.contour, origin='lower', extent=sl.extent, 
                   colors=args.contour_color, linestyles=args.contour_style, linewidths=1)

if args.color:
    print('*** Adding colorbar')
    gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
    gs2 = gridspec.GridSpec(1, 1)
    gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.15, wspace=0.1, hspace=0.1)
    axes = plt.subplot(gs2[0], facecolor='black')
    if args.alpha:
        qi.alphabar(axes, args.color_map, args.color_lims, args.color_label,
                    args.alpha_lims, args.alpha_label,
                    args.contour, args.contour_color, args.contour_style)
    else:
        qi.colorbar(axes, args.color_map, args.color_lims, args.color_label)
else:
    gs1.update(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
print('*** Saving')
print('Writing file: ', args.output)
f.savefig(args.output, facecolor=f.get_facecolor(), edgecolor='none')
plt.close(f)
