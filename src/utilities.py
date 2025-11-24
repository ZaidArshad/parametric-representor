import numpy as np
from scipy.spatial.transform import Rotation as R

def PCA(points):
    """ performs pca on a set of points """
    # move to origin
    mean = np.mean(points, axis=0)
    pts = points - mean

    # calculate covariance and eigenvectors
    covariance_matrix = np.cov(pts, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # sort eigenvectors by eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # ensure right-handed coordinate system because the code freaks out otherwise
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1
    return eigenvectors

def bounding_box(points):
    """ computes the axis-aligned bounding box of a set of points """
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)

    return min_point, max_point

def euler_to_matrix(rx, ry, rz):
    """ converts euler angles to a rotation matrix """
    Rz = R.from_euler('z', rz, degrees=False).as_matrix()
    Ry = R.from_euler('y', ry, degrees=False).as_matrix()
    Rx = R.from_euler('x', rx, degrees=False).as_matrix()
    return Rz @ Ry @ Rx

def matrix_to_euler(R_mat):
    """ converts a rotation matrix to euler angles """
    rot = R.from_matrix(R_mat)
    rx, ry, rz = rot.as_euler('xyz', degrees=False)
    return rx, ry, rz


def distance_to_superquadric(points, x):
    """ computes the implicit distance from points to superquadric """
    R = euler_to_matrix(x[5], x[6], x[7])
    t = x[8:11]

    # transform points (pos and rotation)
    point_c = (points - t) @ R

    # calculate radial distance
    r_norm = np.sqrt(np.sum(point_c ** 2, 1))
    
    # superquadric implicit function
    dist = r_norm * np.abs((
        (((point_c[:, 0] / x[2]) ** 2) ** (1 / x[1]) +
         ((point_c[:, 1] / x[3]) ** 2) ** (1 / x[1])) ** (x[1] / x[0]) +
        ((point_c[:, 2] / x[4]) ** 2) ** (1 / x[0])) ** (-x[0] / 2) - 1
    )
    return dist

def signed_distance_to_superquadric(points, x):
    """ computes the signed implicit distance from points to superquadric """
    R = euler_to_matrix(x[5], x[6], x[7])
    t = x[8:11]

    # transform points (pos and rotation)
    point_c = (points - t) @ R

    # calculating radial distance
    r_norm = np.sqrt(np.sum(point_c ** 2, 1))
    
    # superquadric implicit function
    f = (
        (((point_c[:, 0] / x[2]) ** 2) ** (1 / x[1]) +
         ((point_c[:, 1] / x[3]) ** 2) ** (1 / x[1])) ** (x[1] / x[0]) +
        ((point_c[:, 2] / x[4]) ** 2) ** (1 / x[0])
    ) ** (-x[0] / 2)

    dist = r_norm * (f - 1)
    return dist


def gaussian_3d_from_dist(dist, sigma2):
    """ computes the gaussian probability density from distances in 3D """
    c = (2.0 * np.pi * sigma2) ** (-1.5)
    return c * np.exp(-0.5 * dist**2 / sigma2)

def inlier_probability(distances, sigma, w, p0):
    """ computes the inlier probability for each point """
    sigma2 = sigma**2
    g = gaussian_3d_from_dist(distances, sigma2)
    posterior_inlier = ((1 - w) * g) / ((1 - w) * g + w * p0 + 1e-20)
    return np.clip(posterior_inlier, 1e-12, 1.0 - 1e-12)

def cost_function(x, points, p):
    """ cost function for least squares optimization """
    distances = distance_to_superquadric(points, x)
    weights = np.sqrt(np.clip(p, 1e-12, None))
    residuals = weights * distances
    return residuals