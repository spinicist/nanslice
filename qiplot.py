import numpy as np
import scipy.ndimage.interpolation as ndinterp
import matplotlib as mpl
import matplotlib.cm as cm

def findCorners(img):
    data = img.get_data()

    # Individual axis min/max
    xmin, xmax = np.where(np.any(data, axis=(1, 2)))[0][[0, -1]]
    ymin, ymax = np.where(np.any(data, axis=(0, 2)))[0][[0, -1]]
    zmin, zmax = np.where(np.any(data, axis=(0, 1)))[0][[0, -1]]

    # Now convert to physical space
    corners = np.array([[xmin, ymin, zmin, 1.],
                        [xmax, ymax, zmax, 1.]])
    corners = np.dot(img.get_affine(), corners.T)
    corner1 = np.min(corners[0:3,:], axis=1)
    corner2 = np.max(corners[0:3,:], axis=1)
    # Now do min/max again to standardise corners

    return corner1, corner2

def setupSlice(c1, c2, axis, pos, samples, absolute=False):
    ll = np.copy(c1)
    if (axis == 'z'):
        if absolute:
            ll[2] = pos
        else:
            ll[2] = c1[2]*(1-pos)+c2[2]*pos
        up = np.array([0, c2[1]-c1[1], 0])
        rt = np.array([c2[0]-c1[0], 0, 0])
        extent = (c1[0], c2[0], c1[1], c2[1])
    elif (axis == 'y'):
        if absolute:
            ll[1] = pos
        else:
            ll[1] = c1[1]*(1-pos)+c2[1]*pos
        up = np.array([0, 0, c2[2]-c1[2]])
        rt = np.array([c2[0]-c1[0], 0, 0])
        extent = (c1[0], c2[0], c1[2], c2[2])
    elif (axis == 'x'):
        if absolute:
            ll[0] = pos
        else:
            ll[0] = c1[0]*(1-pos)+c2[0]*pos
        up = np.array([0, c2[1]-c1[1], 0])
        rt = np.array([0, 0, c2[2]-c1[2]])
        extent = (c1[2], c2[2], c1[1], c2[1])
    aspect = np.linalg.norm(up) / np.linalg.norm(rt)
    samples_up = np.round(aspect * samples)
    slice = ll[:, None, None] + (rt[:, None, None] * np.linspace(0, 1, samples)[None, :, None] +
                  up[:, None, None] * np.linspace(0, 1, samples_up)[None, None, :])
    return slice, extent

def sampleSlice(img, sl, order=1):
    old_sz = sl.shape
    new_sz = np.prod(sl.shape[1:])
    sl = sl.reshape([3, new_sz])
    scale = np.mat(img.get_affine()[0:3,0:3]).I
    offset = np.dot(-scale,img.get_affine()[0:3,3]).T
    isl = np.dot(scale, sl) + offset[:]
    isl = np.array(isl).reshape(old_sz)
    return ndinterp.map_coordinates(img.get_data(), isl, order=order).T

def applyCM(data, cm_name, cmin, cmax):
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    cmap = cm.get_cmap(cm_name)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(data,alpha=1,bytes=False)[:,:,0:3]

def scaleAlpha(sl, lo, hi):
    np.clip((sl - lo) / (hi - lo), 0, 1, sl)
    return sl

def blend(sl_under, sl_over, sl_alpha):
    return sl_under*(1 - sl_alpha[:,:,None]) + sl_over*sl_alpha[:,:,None]

def mask(sl, mask):
    return sl*mask[:,:,None]

def alphabar(ax, cm_name, cmin, cmax, clabel, amin, amax, alabel):

    csteps = 128
    asteps = 32

    color = applyCM(np.tile(np.linspace(cmin, cmax, csteps), [asteps, 1]),
                    cm_name, cmin, cmax)
    alpha = np.tile(np.linspace(0, 1, asteps), [csteps, 1]).T
    bg    = np.ones((asteps, csteps, 3))
    acmap = blend(bg, color, alpha)

    ax.imshow(acmap, origin='lower', interpolation='hanning', extent=(cmin,cmax,amin,amax), aspect=1)
    ax.set_xlabel(clabel)
    ax.set_xticks((cmin,(cmin+cmax)/2,cmax))
    ax.set_yticks((amin,amax))
    ax.set_ylabel(alabel)
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')

    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')

    ax.yaxis.label.set_color('w')
    ax.xaxis.label.set_color('w')
    #ax.axis('off')
