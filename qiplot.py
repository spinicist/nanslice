import numpy as np

def findCorners(img):
    data = img.get_data()

    # Individual axis min/max
    xmin, xmax = np.where(np.any(data, axis=(1, 2)))[0][[0, -1]]
    ymin, ymax = np.where(np.any(data, axis=(0, 2)))[0][[0, -1]]
    zmin, zmax = np.where(np.any(data, axis=(0, 1)))[0][[0, -1]]

    # Now convert to physical space
    # Add 0.5 to get voxel centers
    corners = np.array([[xmin+.5, ymin+.5, zmin+.5, 1.],
                        [xmax+.5, ymax+.5, zmax+.5, 1.]])
    print('corners1\n', corners)
    corners = np.dot(img.get_affine(), corners.T)
    print('corners2\n', corners)
    corner1 = np.min(corners[0:3,:], axis=1)
    corner2 = np.max(corners[0:3,:], axis=1)
    print('c1: ', corner1)
    print('c2: ', corner2)
    # Now do min/max again to standardise corners

    return corner1, corner2