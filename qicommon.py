#!/usr/bin/env python
"""qicommon.py

Provides common functions / classes for qiview and qislices"""
import argparse
import numpy as np
import scipy.ndimage.interpolation as ndinterp
import matplotlib as mpl
import matplotlib.cm as cm

def find_bbox(img, padding=0):
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
    corner1 = np.min(corners[0:3, :], axis=1) - padding
    corner2 = np.max(corners[0:3, :], axis=1) + padding
    return corner1, corner2

class Slice:
    """A very simple slice class. Stores physical & voxel space co-ords"""
    def __init__(self, c1, c2, axis, pos, samples, absolute=False):
        axis_map = {'x':0, 'y':1, 'z':2}
        ind_0 = axis_map[axis]
        ind_1 = (ind_0 + 1) % 3
        ind_2 = (ind_0 + 2) % 3
        start = np.copy(c1)
        diag = c2 - c1
        if absolute:
            start[ind_0] = pos
        else:
            start[ind_0] = c1[ind_0]*(1-pos) + c2[ind_0]*pos
        dir_rt = np.zeros((3,))
        dir_up = np.zeros((3,))

        dir_rt[ind_1] = diag[ind_1]
        dir_up[ind_2] = diag[ind_2]
        aspect = np.linalg.norm(dir_up) / np.linalg.norm(dir_rt)
        samples_up = np.round(aspect * samples)
        self._world_space = (start[:, None, None] +
                             dir_rt[:, None, None] * np.linspace(0, 1, samples)[None, :, None] +
                             dir_up[:, None, None] * np.linspace(0, 1, samples_up)[None, None, :])
        # This is the extent parameter for matplotlib
        self.extent = (c1[ind_1], c2[ind_1], c1[ind_2], c2[ind_2])
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

def common_args():
    parser = argparse.ArgumentParser(description='Dual-coding viewer.')
    parser.add_argument('base_image', help='Base (structural image)', type=str)
    parser.add_argument('mask_image', help='Mask image', type=str)
    parser.add_argument('color_image', help='Image for color-coding of overlay', type=str)
    parser.add_argument('alpha_image', help='Image for transparency-coding of overlay', type=str)
    parser.add_argument('--window', type=float, nargs=2, default=(1, 99),
                        help='Specify base image window (in percentiles)')
    parser.add_argument('--samples', type=int, default=128,
                        help='Number of samples for slicing, default=128')
    parser.add_argument('--interp', type=str, default='hanning',
                        help='Display interpolation mode, default=hanning')
    parser.add_argument('--interp_order', type=int, default=1,
                        help='Data interpolation order, default=1')
    parser.add_argument('--alpha_lims', type=float, nargs=2, default=(0.5, 1.0),
                        help='Alpha/transparency window, default=0.5 1.0')
    parser.add_argument('--alpha_label', type=str, default='1-p',
                        help='Label for alpha/transparency axis')
    parser.add_argument('--contour', type=float, default=0.95,
                        help='Specify value for alpha image contour, default=0.95')
    parser.add_argument('--color_lims', type=float, nargs=2, default=(-1, 1),
                        help='Colormap window, default=-1 1')
    parser.add_argument('--color_scale', type=float, default=1,
                        help='Multiply color image by value, default=1')
    parser.add_argument('--color_map', type=str, default='RdYlBu_r',
                        help='Colormap to use from Matplotlib, default = RdYlBu_r')
    parser.add_argument('--color_label', type=str, default='% Change',
                        help='Label for color axis')
    return parser
