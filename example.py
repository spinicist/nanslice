import nibabel as nib
import qiplot

m = nib.load('/Users/Tobias/Data/MATRICS/rat/study_mask.nii')
(corner1, corner2) = qiplot.findCorners(m)

sl = qiplot.setupSlice(corner1, corner2, 'z', 0.5, 64)

sl_img = qiplot.sampleSlice(m, sl)
