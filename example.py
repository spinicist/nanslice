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

f = plt.figure(facecolor='black')
gs1 = gridspec.GridSpec(4, 4)
gs1.update(left=0.01, right=0.99, bottom=0.16, top=0.99, wspace=0.01, hspace=0.01)
gs2 = gridspec.GridSpec(1, 1)
gs2.update(left = 0.08, right = 0.92, bottom = 0.08, top = 0.16, wspace=0.1, hspace=0.1)

for i in range(0,15):
    ax = plt.subplot(gs1[i], axisbg='black')
    (sl, ext) = qiplot.setupSlice(corner1, corner2, 'z', 120 + i, 128, absolute=True)

    sl_mask = qiplot.sampleSlice(mask, sl, order=1)
    sl_pstat = qiplot.scaleAlpha(qiplot.sampleSlice(pstat, sl), 0.5, 1.0)

    sl_template = qiplot.applyCM(qiplot.sampleSlice(template, sl), 'gray', tmin, tmax)
    sl_Tstat = qiplot.applyCM(qiplot.sampleSlice(Tstat, sl), 'RdYlBu', -4, 4)
    sl_blend = qiplot.mask(qiplot.blend(sl_template, sl_Tstat, sl_pstat), sl_mask)

    ax.imshow(sl_blend, origin='lower', extent=ext, interpolation='hanning')
    ax.contour(sl_pstat, (0.95,), origin='lower', extent = ext)
    ax.axis('off')

ax = plt.subplot(gs2[0], axisbg='black')
qiplot.alphabar(ax,'RdYlBu',-4,4,'T-Stat',0.5,1.0,'1-p')
f.savefig('example.png',facecolor=f.get_facecolor(), edgecolor='none')

# TODO: Custom color bars
# TODO: Multiple slices