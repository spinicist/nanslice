import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import qiplot

mask = nib.load('/Users/Tobias/Data/MATRICS/rat/study_mask.nii')
template = nib.load('/Users/Tobias/Data/MATRICS/rat/studytemplate0.nii.gz')
Tstat = nib.load('/Users/Tobias/Data/MATRICS/rat/rlog_jacobian_tstat1.nii.gz')
pstat = nib.load('/Users/Tobias/Data/MATRICS/rat/rlog_jacobian_vox_p_fstat1.nii')
(tmin, tmax) = np.percentile(template.get_data(), (5,99))
(corner1, corner2) = qiplot.findCorners(mask)

(sl, ext) = qiplot.setupSlice(corner1, corner2, 'z', 0.5, 128)

sl_mask = qiplot.sampleSlice(mask, sl, order=1)
sl_pstat = qiplot.scaleAlpha(qiplot.sampleSlice(pstat, sl), 0.5, 1.0)

sl_template = qiplot.applyCM(qiplot.sampleSlice(template, sl), 'gray', tmin, tmax)
sl_Tstat = qiplot.applyCM(qiplot.sampleSlice(Tstat, sl), 'RdYlBu', -4, 4)
sl_blend = qiplot.mask(qiplot.blend(sl_template, sl_Tstat, sl_pstat), sl_mask)

f, ax = plt.subplots(2,1)
ax[0].imshow(sl_blend, origin='lower', extent=ext, interpolation='hanning')
ax[0].contour(sl_pstat, (0.95,), origin='lower', extent = ext)
ax[0].axis('off')
qiplot.alphabar(ax[1],'RdYlBu',-4,4,'T-Stat',0.5,1.0,'1-p')
plt.show()

# TODO: Custom color bars
# TODO: Multiple slices