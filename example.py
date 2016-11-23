import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import qiplot

m = nib.load('/Users/Tobias/Data/MATRICS/rat/study_mask.nii')
t = nib.load('/Users/Tobias/Data/MATRICS/rat/studytemplate0.nii.gz')
(tmin, tmax) = np.percentile(t.get_data(), (5,95))
(corner1, corner2) = qiplot.findCorners(m)

(sl, ext) = qiplot.setupSlice(corner1, corner2, 'z', 0.5, 128)

sl_img = qiplot.sampleSlice(t, sl)
ms_img = qiplot.sampleSlice(m, sl, order=0)
sl_colimg = qiplot.applyCM(sl_img, 'hot', tmin, tmax) * ms_img[:,:,None]
print(sl_colimg[0,0,:])
ax = plt.subplot(axisbg='black')
ax.imshow(sl_colimg, origin='lower', extent=ext, interpolation='hanning')
ax.axis('off')
plt.show()