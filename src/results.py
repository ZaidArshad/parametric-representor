import spherenet
import metrics
import trimesh
import numpy as np
from superquadric import Superquadric
import optimization
import time

def post_results(elapsed_t, pc, gt):
    """
    Test a point cloud to it's ground truth.
    
    Args
        elapsed_t (float): How long it took to generate the point cloud.
        pc (list): Nx3 matrix of fitted point cloud.
        gt (list): Kx3 matrix of ground truth point cloud.
    """
    iou = round(metrics.pc_intersection_over_union(gt, pc), 4)
    chamfer_d = round(metrics.chamfer_distance(gt, pc), 4)
    elapsed_t = round(elapsed_t, 4)
    print(f"Elapsed: {elapsed_t}s, IoU: {iou}, Chamfer Distance: {chamfer_d}")

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

    start_t = time.perf_counter()
    sphere_params = spherenet.determine_sphere_params(gt_surface_points, gt_sdf_points, gt_sdf_values, num_epochs=50)
    sphere_pc = spherenet.sphere_params_to_pc(sphere_params.tolist())
    elapsed_t = time.perf_counter() - start_t 

    if show_visual:
        spherenet.visualise_spheres(sphere_params, reference_model=gt_surface_points)

    post_results(elapsed_t, sphere_pc, gt_surface_points)

def superquadric_test(gt_surface_points, show_visual=True):
    """
    Superquadric fitting test on ground truth point cloud.
    
    Args: 
        gt_surface_points (list): Nx3 matrix of ground truth surface points.
        show_visual (boolean, optional): Whether to show the visual comparison between the fit and ground truth.
    """
    print("Superquadric test.")

    start_t = time.perf_counter()
    args = { 'inlier_ratio': 0.999 }
    superquadric_params, (outliers, inliers) = optimization.points_to_superquadric(gt_surface_points, args=args)
    sq = Superquadric(superquadric_params)
    sq_pc = sq.generate_points()
    elapsed_t = time.perf_counter() - start_t 

    if show_visual:
        mesh = sq.create_mesh(u_res=100, v_res=100)
        pc_outliers_mesh = trimesh.points.PointCloud(outliers, colors=[255, 255, 0, 255])
        pc_inliers_mesh = trimesh.points.PointCloud(inliers, colors=[0, 0, 255, 255])
        scene = trimesh.Scene([mesh, pc_outliers_mesh, pc_inliers_mesh])
        scene.show(background=[0, 0, 0, 255])

    post_results(elapsed_t, sq_pc, gt_surface_points)

if __name__ == "__main__":
    model_name = "dog"

    print(f"Testing {model_name} model.")

    # Ground truth
    gt_surface_points = trimesh.load(f"data/{model_name}/surface_points.ply").vertices
    gt_sdf_model = np.load(f"data/{model_name}/voxel_and_sdf.npz")
    gt_sdf_points = gt_sdf_model["sdf_points"]
    gt_sdf_values = gt_sdf_model["sdf_values"]

    baseline_test(gt_surface_points, gt_sdf_points, gt_sdf_values)
    superquadric_test(gt_surface_points)

    


