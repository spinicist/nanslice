#!python
import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qicommon
 
parser = argparse.ArgumentParser(description='Dual-code a set of slices through an image.')
parser.add_argument('base_image',help='Base (structural image)',type=str)
parser.add_argument('mask_image',help='Mask image',type=str)
parser.add_argument('color_image',help='Image for color-coding of overlay',type=str)
parser.add_argument('alpha_image',help='Image for transparency-coding of overlay',type=str)
parser.add_argument('output',help='Output image name',type=str)
parser.add_argument('--window', nargs=2, default=(1,99), help='Specify base image window (in percentiles)')
parser.add_argument('--alpha_lims', nargs=2,default=(0.5,1.0), help='Alpha/transparency window, default=0.5 1.0')
parser.add_argument('--alpha_label', type=str, default='1-p', help='Label for alpha/transparency axis')
parser.add_argument('--contour',help='Specify value for alpha image contour, default=0.95',type=float,default=0.95)
parser.add_argument('--color_lims', type=float, nargs=2, default=(-1,1), help='Colormap window, default=-1 1')
parser.add_argument('--color_scale', type=float, default=1, help='Multiply color image by value, default=1')
parser.add_argument('--color_map', type=str, default='RdYlBu_r', help='Colormap to use from Matplotlib, default = RdYlBu_r')
parser.add_argument('--color_label', type=str, default='% Change', help='Label for color axis')
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
(corner1, corner2) = qicommon.findCorners(img_mask)
print('Bounding box: ', corner1, ' -> ', corner2)
slice_total = args.slice_rows*args.slice_cols
print(slice_total, ' slices in ', args.slice_rows, ' rows and ', args.slice_cols, ' columns')
slice_pos = np.linspace(args.slice_lims[0], args.slice_lims[1], slice_total)

gs1 = gridspec.GridSpec(args.slice_rows, args.slice_cols)
gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
gs2 = gridspec.GridSpec(1, 1)
gs2.update(left = 0.08, right = 0.92, bottom = 0.08, top = 0.16, wspace=0.1, hspace=0.1)

f = plt.figure(facecolor='black')

print('*** Slicing')
for s in range(0, slice_total):
    ax = plt.subplot(gs1[s], axisbg='black')

    (sl, slice_extent) = qicommon.setupSlice(corner1, corner2, args.slice_axis, slice_pos[s], 128)
    sl_mask = qicommon.sampleSlice(img_mask, sl, order=1)
    sl_base = qicommon.applyCM(qicommon.sampleSlice(img_base, sl), 'gray', window)
    sl_color = qicommon.applyCM(args.color_scale*qicommon.sampleSlice(img_color, sl), args.color_map, args.color_lims)
    sl_alpha = qicommon.sampleSlice(img_alpha, sl)
    sl_blend = qicommon.mask(qicommon.blend(sl_base, sl_color, qicommon.scaleAlpha(sl_alpha, args.alpha_lims)), sl_mask)
    
    ax.imshow(sl_blend, origin='lower', extent = slice_extent, interpolation='hanning')
    ax.axis('off')
    if (args.contour > 0):
        ax.contour(sl_alpha, (args.contour,), origin='lower', extent = slice_extent)

print('*** Saving')
ax = plt.subplot(gs2[0], axisbg='black')
qicommon.alphabar(ax, args.color_map, args.color_lims, args.color_label , args.alpha_lims, args.alpha_label)
print('Writing file: ', args.output)
f.savefig(args.output ,facecolor=f.get_facecolor(), edgecolor='none')
plt.close(f)