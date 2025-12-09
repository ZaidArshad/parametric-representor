import os
import sys
import numpy as np
import pandas as pd
import trimesh
import time
import argparse
from random import randint
from skopt import gp_minimize
from skopt.space import Real, Integer
import optimization
import metrics
from superquadric import Superquadric

def run_test(points, inlier_ratio, switching_threshold, min_cluster_size, n_clusters):
    """
    Runs the superquadric fitting process for a given set of points and parameters, 
    and returns the IoU and Chamfer Distance.
    """
    # points is the ground truth for this test
    gt_surface_points = points

    args = {
        'inlier_ratio': inlier_ratio,
        'switching_threshold': switching_threshold,
        'min_cluster_size': int(min_cluster_size),
        'n_clusters': int(n_clusters),
        'iterations': 10,
    }

    # run the fitting
    try:
         superquadric_params, clusters = optimization.points_to_superquadrics(gt_surface_points, args=args)
    except Exception as e:
        print(f"Error during optimization: {e}")
        return None, None
    
    superquadrics = [Superquadric(x) for x in superquadric_params]
    
    if not superquadrics:
        # there were no superquadrics fitted
        return 0.0, float('inf')

    # generate point clouds from fitted superquadrics
    superquadric_points = np.concatenate([sq.generate_points() for sq in superquadrics])
    
    # calculate metrics
    iou = metrics.pc_intersection_over_union(gt_surface_points, superquadric_points)
    chamfer = metrics.chamfer_distance(gt_surface_points, superquadric_points)
    
    return iou, chamfer

def objective_function(params, points):
    """
    The objective function for Bayesian optimization
    """
    inlier_ratio, switching_threshold, min_cluster_size, n_clusters = params
    
    # run the test
    iou, chamfer = run_test(points, inlier_ratio, switching_threshold, min_cluster_size, n_clusters)
    
    iou_str = f"{iou:.4f}" if iou is not None else "Failed"
    print(f"    params: inlier ratio={inlier_ratio:.4f}, switching threshold={switching_threshold:.6f}, min cluster size={min_cluster_size}, n_clusters={n_clusters}. IoU: {iou_str}")

    if iou is None:
        # penalize failed runs
        return 100 # completely arbitrary number here
        
    # return negative iou for minimization
    return -iou 

def optimize_superquadrics(points):
    print(f"optimizing metaparams...")
    
    # search space
    space = [
        # continuous inlier ratio
        Real(0.7, 1.0 - 1e-6, name='inlier_ratio'), 
        
        # continuous switching threshold
        Real(1e-6, 0.1, name='switching_threshold'), 
        
        # discrete: min cluster size
        Integer(1, 100, name='min_cluster_size'),

        # if we add more parameters, we can add them here

        Integer(1, 100, name='n_clusters'),
    ]
    
    start_time = time.time()
    
    # we need to freeze the points for the objective function
    def objective(params):
        return objective_function(params, points)

    # run the optimization
    res_bo = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=15,          # total number of evaluations of the objective function
        n_initial_points=8, # this many random initial points to sample
        acq_func="gp_hedge",
        verbose=False
    )
    
    elapsed_time = time.time() - start_time
    print(f"Finished optimization in {elapsed_time:.2f} seconds.")

    # extract best results
    best_params = res_bo.x
    
    # run the final best test to get the chamfer distance and display metrics
    final_iou, final_chamfer = run_test(points, best_params[0], best_params[1], best_params[2], best_params[3])

    print(f"\nbest parameters found:")
    print(f"  inlier ratio: {best_params[0]:.4f}")
    print(f"  switching threshold: {best_params[1]:.6f}")
    print(f"  min cluster size: {int(best_params[2])}")
    print(f"  n clusters: {int(best_params[3])}")
    print(f"  IoU: {final_iou:.4f}")
    print(f"  chamfer distance: {final_chamfer:.4f}")
    
    best_args = {
        'inlier_ratio': best_params[0],
        'switching_threshold': best_params[1],
        'min_cluster_size': int(best_params[2]),
        'n_clusters': int(best_params[3]),
        'iterations': 10,
    }

    return optimization.points_to_superquadrics(points, args=best_args)
