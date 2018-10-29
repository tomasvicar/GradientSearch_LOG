from numpy.linalg import norm
from math import pi
from scipy.ndimage.filters import gaussian_laplace, minimum_filter
from operator import contains
from functools import partial
from itertools import filterfalse
from skimage import filters
from numpy import copy, cross, dot

def findBlobs(img, datatype, z_x, y_x, mode, scalerange, max_overlap=0.05):
    from numpy import ones, triu, seterr, sqrt
    old_errs = seterr(invalid='ignore')
    peaks = blobLOG(img, z_x, y_x, mode, scales=scalerange)
    if datatype=='2D':
        radii = sqrt(2)*peaks[:, 0]
    elif datatype=='3D':
        radii=sqrt(3)*peaks[:, 0]
        
    positions = copy(peaks[:, 1:])
    positions[:, 0]=z_x*positions[:,0]
    positions[:, 1]=y_x*positions[:,1]
    if mode == 'conservative':
        distances = norm(positions[:, None, :] - positions[None, :, :], axis=2)
        # mode
        if datatype=='2D':
            intersections = circleIntersection(radii, radii.T, distances)
            volumes = pi * radii ** 2
        elif datatype=='3D':
            intersections = sphereIntersection(radii, radii.T, distances)
            volumes = 4/3 * pi * radii ** 3
        else:
            raise ValueError("Invalid dimensions for position ({}), need 2 or 3."
                             .format(positions.shape[1]))

        delete = ((intersections > (volumes * max_overlap))
                  # Remove the smaller of the blobs
                  & ((radii[:, None] < radii[None, :])
                     # Tie-break
                     | ((radii[:, None] == radii[None, :])
                        & triu(ones((len(peaks), len(peaks)), dtype='bool'))))
        ).any(axis=1)
        seterr(**old_errs)
        return peaks[~delete]
    elif mode == 'liberal':
        return peaks

def blobLOG(data, z_x, y_x, mode, scales=range(1, 10, 1)):
    """Find blobs. Returns [[scale, z, y, x,...], ...]"""
    from numpy import empty, asarray
    from itertools import repeat

    data = asarray(data)
    scales = asarray(scales)

    log = empty((len(scales),) + data.shape, dtype=data.dtype)
    for slog, scale in zip(log, scales):
        if data.ndim==2:
            slog[...] = scale ** 2 * gaussian_laplace(data, [scale/y_x, scale])
        elif data.ndim==3:
            slog[...] = scale ** 2 * gaussian_laplace(data, [scale/z_x, scale/y_x, scale])
            
    peaks = localMinima(log, mode)
    peaks[:, 0] = scales[peaks[:, 0]]
    return peaks

def localMinima(data, mode):
    from numpy import ones, nonzero, transpose
    threshold = filters.threshold_otsu(data)
    if mode =='liberal':
        threshold =0.1*threshold
    if threshold is not None:
        peaks = data < threshold
    else:
        peaks = ones(data.shape, dtype=data.dtype)
    
    peaks &= data == minimum_filter(data, size=(3,) * data.ndim)
    return transpose(nonzero(peaks))

def sphereIntersection(r1, r2, d):
    # https://en.wikipedia.org/wiki/Spherical_cap#Application

    valid = (d < (r1 + r2)) & (d > 0)
    
    return (pi * (r1 + r2 - d) ** 2
            * (d ** 2 
               + 2 * d * (r1 + r2)
               - 3 * (r1 - r2) ** 2)
            / (12 * d)) * valid

def circleIntersection(r1, r2, d):
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    from numpy import arccos, sqrt

    return (r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            + r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            - sqrt((-d + r1 + r2) * (d + r1 - r2)
                   * (d - r1 + r2) * (d + r1 + r2)) / 2)


def getTangentialPlane(threeblobs):
    b1, b2, b3= threeblobs
    
    v1 = b3 - b1
    v2 = b2 - b1

    # the cross product is a vector normal to the plane
    cp = cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = dot(cp, b3)
    a=a/d
    b=b/d
    c=c/d
    d=1
        
    return a,b,c
        
        
