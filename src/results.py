import spherenet
import metrics
import trimesh
import argparse
import numpy as np
from random import randint
from superquadric import Superquadric
import optimization

def test_pc_to_gt(pc, gt):
    """
    Test a point cloud to it's ground truth.

    Args
        pc (list): Nx3 matrix of fitted point cloud
        gt (list): Kx3 matrix of ground truth point cloud
    """
    iou = metrics.pc_intersection_over_union(gt, pc)
    chamfer_d = metrics.chamfer_distance(gt, pc)
    print(f"Intersection over Union: {iou}, Chamfer Distance: {chamfer_d}")

def baseline_test(gt_surface_points, gt_sdf_points, gt_sdf_values, show_visual=True):
    """
    Baseline fitting test using SphereNet on ground truth.

    Args:
        gt_surface_points (list): Nx3 matrix of ground truth surface points.
        gt_sdf_points (list): Kx3 matrix of ground truth sdf points.
        gt_sdf_values (list): List of ground truth K signed distances.
        show_visual (boolean, optional): Whether to show the visual comparison between the fit and ground truth.
    """
    print("Baseline test.")

    sphere_params = spherenet.determine_sphere_params(gt_surface_points, gt_sdf_points, gt_sdf_values, num_epochs=50)
    sphere_pc = spherenet.sphere_params_to_pc(sphere_params.tolist())

    if show_visual:
        spherenet.visualise_spheres(sphere_params, reference_model=gt_surface_points)

    test_pc_to_gt(sphere_pc, gt_surface_points)

def superquadric_test(gt_surface_points, show_visual=True):
    """
    Superquadric fitting test on ground truth point cloud.

    Args:
        gt_surface_points (list): Nx3 matrix of ground truth surface points.
        show_visual (boolean, optional): Whether to show the visual comparison between the fit and ground truth.
    """
    print("Superquadric test.")

    # initialize the arguments to the fitting algorithm
    args = {
            'inlier_ratio': 0.99,
            'switching_threshold': 0.005,
            'min_cluster_size': 5,
            'iterations': 3,
            }

    # get the optimized superquadrics
    superquadric_params, clusters = optimization.points_to_superquadrics(gt_surface_points, args=args)

    superquadrics = [Superquadric(x) for x in superquadric_params]

    # visualize fitted superquadrics
    if show_visual:
        colors = [[randint(0, 255), randint(0, 255), randint(0, 255), 127] for _ in clusters]
        sq_meshes = [superquadrics[i].create_mesh(20, 20, colors[i]) for i in range(len(superquadrics))]
        pc_meshes = [trimesh.points.PointCloud(clusters[i], colors=colors[i]) for i in range(len(clusters))]

        scene = trimesh.Scene([*sq_meshes, *pc_meshes])
        scene.show(background=[0, 0, 0, 255])

    # get a combined point cloud of all superquadrics
    superquadric_points = np.concatenate([sq.generate_points() for sq in superquadrics])
    test_pc_to_gt(superquadric_points, gt_surface_points)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    args = parser.parse_args()

    model_name = args.model_name

    print(f"Testing {model_name} model.")

    # Ground truth
    gt_surface_points = trimesh.load(f"data/{model_name}/surface_points.ply").vertices
    gt_sdf_model = np.load(f"data/{model_name}/voxel_and_sdf.npz")
    gt_sdf_points = gt_sdf_model["sdf_points"]
    gt_sdf_values = gt_sdf_model["sdf_values"]

    baseline_test(gt_surface_points, gt_sdf_points, gt_sdf_values)
    superquadric_test(gt_surface_points)
