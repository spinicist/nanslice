import numpy as np
import scipy.ndimage.interpolation as ndinterp
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

def setupSlice(c1, c2, axis, f, samples):

    # xs = np.linspace(corner1[0], corner2[0], samples)
    # ys = np.linspace(corner1[1], corner2[1], samples)
    # zx = np.linspace(corner1[2], corner2[2], samples)

    # if (axis == 'x'):
    #     xs = (corner1[0]*(1-position) + corner2[0]*position)
    # else if (axis == 'y'):
    #     ys = (corner1[1]*(1-position) + corner2[1]*position)
    # else if (axis == 'z'):
    #     zs = (corner1[2]*(1-position) + corner2[2]*position)

    if (axis == 'z'):
        ll = np.array([c1[0], c1[1], c1[2]*(1-f)+c2[2]*f])
        up = np.array([0, c2[1]-c1[1], 0])
        rt = np.array([c2[0]-c1[0], 0, 0])
    
    slice = ll[:, None, None] + (rt[:, None, None] * np.linspace(0, 1, samples)[None, :, None] +
                  up[:, None, None] * np.linspace(0, 1, samples)[None, None, :])
    
    return slice

def sampleSlice(img, sl):
    old_sz = sl.shape
    new_sz = np.prod(sl.shape[1:])
    sl = sl.reshape([3, new_sz])
    scale = np.mat(img.get_affine()[0:3,0:3]).I
    offset = np.dot(-scale,img.get_affine()[0:3,3]).T
    isl = np.dot(scale, sl) + offset[:]
    isl = np.array(isl).reshape(old_sz)
    return ndinterp.map_coordinates(img.get_data(), isl)