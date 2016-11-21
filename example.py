import nibabel as nib
import qiplot

m = nib.load('/Users/Tobias/Data/MATRICS/mouse/mask.nii')
(corner1, corner2) = qiplot.findCorners(m)