import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qiplot

print('Loading')
mask = nib.load('/Users/Tobias/Data/MATRICS/mouse/mask.nii')
template = nib.load('/Users/Tobias/Data/MATRICS/mouse/stdWarped.nii.gz')
labels = nib.load('/Users/Tobias/Data/MATRICS/mouse/c57_fixed_labels_resized.nii')
Tstat = nib.load('/Users/Tobias/Data/MATRICS/mouse/rlog_jacobian_tstat1.nii.gz')
pstat = nib.load('/Users/Tobias/Data/MATRICS/mouse/rlog_jacobian_vox_p_fstat1.nii')
print('Setup')
(tmin, tmax) = np.percentile(template.get_data(), (1,99))
(corner1, corner2) = qiplot.findCorners(mask)

f = plt.figure(facecolor='black')

nrows = 4
ncols = 5
ntotal = nrows*ncols

sel_labels = ((6,'k'),(106,'k'),(180,'g'),(181,'g'))
cmap = 'RdYlBu_r'

gs1 = gridspec.GridSpec(nrows, ncols)
gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
gs2 = gridspec.GridSpec(1, 1)
gs2.update(left = 0.08, right = 0.92, bottom = 0.08, top = 0.16, wspace=0.1, hspace=0.1)

for i in range(0,ntotal):
    print(i)
    ax = plt.subplot(gs1[i], axisbg='black')
    print('Setup slice')
    (sl, ext) = qiplot.setupSlice(corner1, corner2, 'y', i / (ntotal - 1), 128)
    print('Sample')
    sl_mask = qiplot.sampleSlice(mask, sl, order=1)
    sl_pstat = qiplot.scaleAlpha(qiplot.sampleSlice(pstat, sl), 0.5, 1.0)
    sl_template = qiplot.applyCM(qiplot.sampleSlice(template, sl), 'gray', tmin, tmax)
    sl_Tstat = qiplot.applyCM(qiplot.sampleSlice(Tstat, sl), cmap, -4, 4)
    sl_lbls = qiplot.sampleSlice(labels, sl, order=0)
    print('Blend')
    sl_blend = qiplot.mask(qiplot.blend(sl_template, sl_Tstat, sl_pstat), sl_mask)
    print('Plot')
    ax.imshow(sl_blend, origin='lower', extent=ext, interpolation='hanning')
    for l in sel_labels:
        this_label = (sl_lbls == l[0])
        ax.contour(this_label, (0.5,), origin='lower', extent = ext, colors = l[1])
    ax.axis('off')
print('Colorbar')
ax = plt.subplot(gs2[0], axisbg='black')
qiplot.alphabar(ax,cmap,-4,4,'T-Stat',0.5,1.0,'1-p')
f.savefig('example_blend.png',facecolor=f.get_facecolor(), edgecolor='none')