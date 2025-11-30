import numpy as np
from utilities import *
from superquadric import Superquadric
import scipy
import time

def points_to_superquadric(points, args=None):
    """ 
    optimizes superquadric parameters to fit a set of 3D points 
    
    inputs:
        points: Nx3 array of 3D points
        args: dictionary of arguments 
        - inlier_ratio: expected ratio of inliers in the point cloud

    outputs:
        x: optimized superquadric parameters
        outliers: Mx3 array of outlier points
        inliers: Kx3 array of inlier points
    """
    # arguments
    inlier_ratio = 0.9 if args is None else args["inlier_ratio"]

    # compute centroid and center points
    centroid = np.mean(points, axis=0)
    points = points - centroid

    # normalize points
    max_dist = np.max(np.abs(points))
    scale = max_dist / 10.0
    points = points / scale

    # initial rotation using PCA 
    R_init = PCA(points)
    initial_rotation = matrix_to_euler(R_init)

    # initial scale using rotated bounding box
    rotated_points = points @ R_init
    bbox_min, bbox_max = bounding_box(rotated_points)
    initial_scale = (bbox_max - bbox_min) / 5.0

    # initial superquadric parameters -> x: [e1, e2, a1, a2, a3, rx, ry, rz, tx, ty, tz]
    x0 = np.array([
                    1.0, 1.0,                                                        # e1, e2                                            
                    initial_scale[0], initial_scale[1], initial_scale[2],            # a1, a2, a3
                    initial_rotation[0], initial_rotation[1], initial_rotation[2],   # rx, ry, rz
                    0.0, 0.0, 0.0                                                    # tx, ty, tz 
            ])

    # define lower and upper bounds for parameters
    upper = 4 * np.max(np.abs(points))

    lower_bounds = np.array([
        0.1, 0.1,                         # e1, e2
        0.001, 0.001, 0.001,              # a1, a2, a3
        -np.pi, -np.pi, -np.pi,           # rx, ry, rz
        -upper, -upper, -upper            # tx, ty, tz
    ])

    upper_bounds = np.array([
        2.0, 2.0,                         # e1, e2
        upper, upper, upper,              # a1, a2, a3
        np.pi, np.pi, np.pi,              # rx, ry, rz
        upper, upper, upper               # tx, ty, tz
    ])

    # calculate bounding volume
    bbox_min, bbox_max = bounding_box(points)
    V = np.prod(bbox_max - bbox_min)

    # calculate prior outlier density (p0 = 1/V)
    p0 = 1.0 / V

    # calculate sigma (noise parameter)
    sigma = V**(1/3) / 10.0

    # initialize EMS loop
    x = np.minimum(np.maximum(x0, lower_bounds), upper_bounds)
    p = np.ones(points.shape[0])
    previous_cost = np.inf

    start_time = time.time()
    for iteration in range(20):
        #print(f"\nIteration {iteration+1}")
        iter_start = time.time()

        # E step
        dists = distance_to_superquadric(points, x)
        p = inlier_probability(dists, sigma, inlier_ratio, p0)

        # M step
        optfunc = scipy.optimize.least_squares(
            fun=cost_function,
            x0=x,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=5000,
            args=(points, p)
        )

        x_new = optfunc.x

        # S step
        if previous_cost == np.inf: cost_change = -1
        else: cost_change = (optfunc.cost - previous_cost) / previous_cost

        previous_cost = optfunc.cost

        # when the solution stops improving, find similar superquadrics
        if cost_change > -0.01:
            similars = Superquadric(x_new).get_similars()

            best_candidate = x_new
            best_p = p
            best_cost = optfunc.cost

            for candidate in similars:

                candidate_x = np.minimum(np.maximum(candidate.x, lower_bounds), upper_bounds)

                # E step
                candidate_dists = distance_to_superquadric(points, candidate_x)
                candidate_p = inlier_probability(candidate_dists, sigma, inlier_ratio, p0)

                # M step
                candidate_optfunc = scipy.optimize.least_squares(
                        fun=cost_function,
                        x0=candidate_x,
                        bounds=(lower_bounds, upper_bounds),
                        max_nfev=5000,
                        args=(points, candidate_p)
                        )

                if candidate_optfunc.cost < best_cost:
                    best_candidate = candidate_x
                    best_p = candidate_p
                    best_cost = candidate_optfunc.cost

            x_new = best_candidate
            p = best_p

        # calculate sigma
        dists_new = distance_to_superquadric(points, x_new)
        weights_new = p
        sigma_new = np.sqrt(np.sum(weights_new * (dists_new ** 2)) / (3.0 * np.sum(weights_new)))

        sigma = sigma_new
        x = x_new

        #print(f"  iteration time: {(time.time() - iter_start):.3f}s")

    print(f"Cluster optimization time: {(time.time() - start_time):.3f}s")

    # fix translation and scale
    x[8:11] = x[8:11] * scale + centroid
    x[2:5] = x[2:5] * scale

    # generate outliers (distance > threshold and inside superquadric)
    points_original = points * scale + centroid
    dists = distance_to_superquadric(points_original, x)
    signed_dists = signed_distance_to_superquadric(points_original, x)
    outlier_mask = (dists > 0.03) & (signed_dists < 0)
    outliers = points_original[outlier_mask]
    inliers = points_original[~outlier_mask]

    return x, (outliers, inliers)
