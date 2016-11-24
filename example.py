import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qiplot

mask = nib.load('/Users/Tobias/Data/MATRICS/rat/study_mask.nii')
template = nib.load('/Users/Tobias/Data/MATRICS/rat/studytemplate0.nii.gz')
Tstat = nib.load('/Users/Tobias/Data/MATRICS/rat/rlog_jacobian_tstat1.nii.gz')
pstat = nib.load('/Users/Tobias/Data/MATRICS/rat/rlog_jacobian_vox_p_fstat1.nii')
(tmin, tmax) = np.percentile(template.get_data(), (5,99))
(corner1, corner2) = qiplot.findCorners(mask)

gs = gridspec.GridSpec(4, 4, height_ratios=[1,1,1,0.5])
gs.update(left=0.01, right=0.99, bottom=0.1, top=0.99, wspace=0.01, hspace=0.01)
f = plt.figure(facecolor='black')

for i in range(0,15):
    ax = plt.subplot(gs[i], axisbg='black')
    (sl, ext) = qiplot.setupSlice(corner1, corner2, 'z', (i / 15), 128)

    sl_mask = qiplot.sampleSlice(mask, sl, order=1)
    sl_pstat = qiplot.scaleAlpha(qiplot.sampleSlice(pstat, sl), 0.5, 1.0)

    sl_template = qiplot.applyCM(qiplot.sampleSlice(template, sl), 'gray', tmin, tmax)
    sl_Tstat = qiplot.applyCM(qiplot.sampleSlice(Tstat, sl), 'RdYlBu', -4, 4)
    sl_blend = qiplot.mask(qiplot.blend(sl_template, sl_Tstat, sl_pstat), sl_mask)

    ax.imshow(sl_blend, origin='lower', extent=ext, interpolation='hanning')
    ax.contour(sl_pstat, (0.95,), origin='lower', extent = ext)
    ax.axis('off')

gs = gridspec.GridSpec(1, 1)
gs.update(left = 0.05, right = 0.95, bottom = 0.05, top = 0.1, wspace=0, hspace=0)
ax = plt.subplot(gs[0], axisbg='black')
qiplot.alphabar(ax,'RdYlBu',-4,4,'T-Stat',0.5,1.0,'1-p')
f.savefig('example.png',facecolor=f.get_facecolor(), edgecolor='none')

# TODO: Custom color bars
# TODO: Multiple slices