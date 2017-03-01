"""qicommon.py

Provides common functions / classes for qiview and qislices"""
import numpy as np
import scipy.ndimage.interpolation as ndinterp
import matplotlib as mpl
import matplotlib.cm as cm

def find_bbox(img):
    """Finds the bounding box of non-zero voxels"""
    data = img.get_data()

    # Individual axis min/maxes
    xmin, xmax = np.where(np.any(data, axis=(1, 2)))[0][[0, -1]]
    ymin, ymax = np.where(np.any(data, axis=(0, 2)))[0][[0, -1]]
    zmin, zmax = np.where(np.any(data, axis=(0, 1)))[0][[0, -1]]

    # Convedir_rt to physical space
    corners = np.array([[xmin, ymin, zmin, 1.],
                        [xmax, ymax, zmax, 1.]])
    corners = np.dot(img.get_affine(), corners.T)
    # Now do min/maxes again to standardise corners
    corner1 = np.min(corners[0:3, :], axis=1)
    corner2 = np.max(corners[0:3, :], axis=1)
    return corner1, corner2

class Slice:
    """A very simple slice class. Stores physical & voxel space co-ords"""
    def __init__(self, c1, c2, axis, pos, samples, absolute=False):
        start = np.copy(c1)
        if axis == 'z':
            if absolute:
                start[2] = pos
            else:
                start[2] = c1[2]*(1-pos)+c2[2]*pos
            dir_up = np.array([0, c2[1]-c1[1], 0])
            dir_rt = np.array([c2[0]-c1[0], 0, 0])
            extent = (c1[0], c2[0], c1[1], c2[1])
        elif axis == 'y':
            if absolute:
                start[1] = pos
            else:
                start[1] = c1[1]*(1-pos)+c2[1]*pos
            dir_up = np.array([0, 0, c2[2]-c1[2]])
            dir_rt = np.array([c2[0]-c1[0], 0, 0])
            extent = (c1[0], c2[0], c1[2], c2[2])
        elif axis == 'x':
            if absolute:
                start[0] = pos
            else:
                start[0] = c1[0]*(1-pos)+c2[0]*pos
            dir_up = np.array([0, 0, c2[2]-c1[2]])
            dir_rt = np.array([0, c2[1]-c1[1], 0])
            extent = (c1[1], c2[1], c1[2], c2[2])
        aspect = np.linalg.norm(dir_up) / np.linalg.norm(dir_rt)
        samples_up = np.round(aspect * samples)
        self._world_space = (start[:, None, None] +
                             dir_rt[:, None, None] * np.linspace(0, 1, samples)[None, :, None] +
                             dir_up[:, None, None] * np.linspace(0, 1, samples_up)[None, None, :])
        self.extent = extent
        self._tfm = None
        self._voxel_space = None

    def get_physical(self, tfm):
        """Returns an array of physical space co-ords for this slice. Will be cached."""
        if not np.array_equal(tfm, self._tfm):
            old_sz = self._world_space.shape
            new_sz = np.prod(self._world_space.shape[1:])
            scale = np.mat(tfm[0:3, 0:3]).I
            offset = np.dot(-scale, tfm[0:3, 3]).T
            isl = np.dot(scale, self._world_space.reshape([3, new_sz])) + offset[:]
            isl = np.array(isl).reshape(old_sz)
            self._voxel_space = isl
            self._tfm = tfm
        return self._voxel_space

def sample_point(img, point, order=1):
    scale = np.mat(img.get_affine()[0:3, 0:3]).I
    offset = np.dot(-scale, img.get_affine()[0:3, 3]).T
    s_point = np.dot(scale, point).T + offset[:]
    return ndinterp.map_coordinates(img.get_data().squeeze(), s_point, order=order)

def sample_slice(img, img_slice, order=1):
    physical = img_slice.get_physical(img.get_affine())
    return ndinterp.map_coordinates(img.get_data().squeeze(), physical, order=order).T

def apply_color(data, cm_name, clims):
    norm = mpl.colors.Normalize(vmin=clims[0], vmax=clims[1])
    cmap = cm.get_cmap(cm_name)
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    return smap.to_rgba(data, alpha=1, bytes=False)[:, :, 0:3]

def scale_clip(data, lims):
    return np.clip((data - lims[0]) / (lims[1] - lims[0]), 0, 1)

def blend_imgs(img_under, img_over, img_alpha):
    return img_under*(1 - img_alpha[:, :, None]) + img_over*img_alpha[:, :, None]

def mask_img(img, img_mask, back=np.array((0, 0, 0))):
    return blend_imgs(back, img, img_mask)

def alphabar(axes, cm_name, clims, clabel, alims, alabel, black_backg=True):
    """Plots a 2D colorbar (color/alpha)"""
    csteps = 64
    asteps = 32
    color = apply_color(np.tile(np.linspace(clims[0], clims[1], csteps),
                                [asteps, 1]), cm_name, clims)
    alpha = np.tile(np.linspace(0, 1, asteps), [csteps, 1]).T
    backg = np.ones((asteps, csteps, 3))
    acmap = blend_imgs(backg, color, alpha)

    axes.imshow(acmap, origin='lower', interpolation='hanning',
                extent=(clims[0], clims[1], alims[0], alims[1]),
                aspect='auto')
    axes.set_xticks((clims[0], np.sum(clims)/2, clims[1]))
    axes.set_xticklabels(('{:.1f}'.format(clims[0]),
                          clabel,
                          '{:.1f}'.format(clims[1])))
    axes.set_yticks((alims[0], np.sum(alims)/2, alims[1]))
    axes.set_yticklabels(('{:.1f}'.format(alims[0]),
                          alabel,
                          '{:.1f}'.format(alims[1])))
    if black_backg:
        axes.spines['bottom'].set_color('w')
        axes.spines['top'].set_color('w')
        axes.spines['right'].set_color('w')
        axes.spines['left'].set_color('w')
        axes.tick_params(axis='x', colors='w')
        axes.tick_params(axis='y', colors='w')
        axes.yaxis.label.set_color('w')
        axes.xaxis.label.set_color('w')
    else:
        axes.spines['bottom'].set_color('k')
        axes.spines['top'].set_color('k')
        axes.spines['right'].set_color('k')
        axes.spines['left'].set_color('k')
        axes.tick_params(axis='x', colors='k')
        axes.tick_params(axis='y', colors='k')
        axes.yaxis.label.set_color('k')
        axes.xaxis.label.set_color('k')
        axes.axis('on')
