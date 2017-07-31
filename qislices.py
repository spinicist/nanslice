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
args = parser.parse_args()

print('*** Loading files')
print('Loading: ', args.base_image)
img_base = nib.load(args.base_image)
print('Loading: ', args.mask_image)
img_mask = nib.load(args.mask_image)
print('Loading: ', args.color_image)
img_color = nib.load(args.color_image)
print('Loading: ', args.alpha_image)
img_alpha = nib.load(args.alpha_image)

print('*** Setup')
window = np.percentile(img_base.get_data(), args.window)
print('Base image window: ', window[0], ' - ', window[1])
(corner1, corner2) = qi.find_bbox(img_mask)
print('Bounding box: ', corner1, ' -> ', corner2)
slice_total = args.slice_rows*args.slice_cols
print(slice_total, ' slices in ', args.slice_rows, ' rows and ', args.slice_cols, ' columns')
slice_pos = np.linspace(args.slice_lims[0], args.slice_lims[1], slice_total)

gs1 = gridspec.GridSpec(args.slice_rows, args.slice_cols)
gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
gs2 = gridspec.GridSpec(1, 1)
gs2.update(left=0.08, right=0.92, bottom=0.08, top=0.15, wspace=0.1, hspace=0.1)

f = plt.figure(facecolor='black')

print('*** Slicing')
for s in range(0, slice_total):
    ax = plt.subplot(gs1[s], facecolor='black')
    print('Slice pos ', slice_pos[s])
    sl = qi.Slice(corner1, corner2, args.slice_axis, slice_pos[s], args.samples)
    sl_mask = qi.sample_slice(img_mask, sl, args.interp_order)
    sl_base = qi.apply_color(qi.sample_slice(img_base, sl, order=args.interp_order), 'gray', window)
    sl_color = qi.apply_color(args.color_scale *
                              qi.sample_slice(img_color, sl, order=args.interp_order),
                              args.color_map, args.color_lims)
    sl_alpha = qi.sample_slice(img_alpha, sl, order=args.interp_order)
    sl_blend = qi.blend_imgs(sl_base, sl_color,
                             qi.scale_clip(sl_alpha, args.alpha_lims))
    sl_masked = qi.mask_img(sl_blend, sl_mask)

    ax.imshow(sl_masked, origin='lower', extent=sl.extent, interpolation=args.interp)
    ax.axis('off')
    if args.contour > 0:
        ax.contour(sl_alpha, levels=[args.contour], origin='lower', extent=sl.extent)

print('*** Saving')
axes = plt.subplot(gs2[0], facecolor='black')
qi.alphabar(axes, args.color_map, args.color_lims, args.color_label,
            args.alpha_lims, args.alpha_label)
print('Writing file: ', args.output)
f.savefig(args.output, facecolor=f.get_facecolor(), edgecolor='none')
plt.close(f)
