import spherenet
import metrics
import trimesh
import os
import numpy as np
from random import randint
from superquadric import Superquadric
import optimization
import time
import sys
import csv
import parameter_search

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
    return (elapsed_t, chamfer_d, iou)

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

    return post_results(elapsed_t, sphere_pc, gt_surface_points)

def superquadric_test(gt_surface_points, show_visual=True):
    """
    Superquadric fitting test on ground truth point cloud.

    Args:
        gt_surface_points (list): Nx3 matrix of ground truth surface points.
        show_visual (boolean, optional): Whether to show the visual comparison between the fit and ground truth.
    """
    print("Superquadric test.")

    start_t = time.perf_counter()

    # get the optimized superquadrics
    superquadric_params, clusters = parameter_search.optimize_superquadrics(gt_surface_points)
    superquadrics = [Superquadric(x) for x in superquadric_params]

    elapsed_t = time.perf_counter() - start_t 

    # visualize fitted superquadrics
    if show_visual:
        colors = [[randint(0, 255), randint(0, 255), randint(0, 255), 255] for _ in clusters]
        sq_meshes = [superquadrics[i].create_mesh(20, 20, colors[i]) for i in range(len(superquadrics))]
        #pc_meshes = [trimesh.points.PointCloud(clusters[i], colors=colors[i]) for i in range(len(clusters))]
        gt_mesh = trimesh.points.PointCloud(gt_surface_points, [255, 127, 127, 127])

        scene = trimesh.Scene([*sq_meshes, gt_mesh])
        scene.show(background=[255, 255, 255, 255])

    # get a combined point cloud of all superquadrics
    superquadric_points = np.concatenate([sq.generate_points() for sq in superquadrics])
    return post_results(elapsed_t, superquadric_points, gt_surface_points)

def save_results(file_name, results):
    print(f"Saving to {file_name}")
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)

def summarize_results():
    summary_file_name = "results/summary.csv"
    if (os.path.exists(summary_file_name)):
        os.remove(summary_file_name)
    summaries = []
    for file_name in os.listdir("results"):
        with open(f"results/{file_name}", "r") as file:
            reader = csv.reader(file)
            rows = np.array([[float(s) for s in row] for row in reader])
            summary = (file_name, np.mean(rows, axis=0))
            summaries.append(summary)
    save_results(summary_file_name, summaries)


def run_tests(test_type, save_to_file, iterations, show_visual, model_name):
        # Ground truth
        gt_surface_points = trimesh.load(f"data/{model_name}/surface_points.ply").vertices
        gt_sdf_model = np.load(f"data/{model_name}/voxel_and_sdf.npz")
        gt_sdf_points = gt_sdf_model["sdf_points"]
        gt_sdf_values = gt_sdf_model["sdf_values"]

        results = []
        for i in range(iterations): 
            result = (baseline_test(gt_surface_points, gt_sdf_points, gt_sdf_values, show_visual) if test_type == "base" 
                      else superquadric_test(gt_surface_points, show_visual))
            results.append(result)
        if (save_to_file):
            output_file = f"results/result_{test_type}_{model_name}.csv"
            save_results(output_file, results)

if __name__ == "__main__":
    # Example python src/results.py all 1 0 5 dog
    test_type = sys.argv[1] # all, base or superquadric
    save_to_file = bool(int(sys.argv[2])) # 0 or 1
    show_visual = bool(int(sys.argv[3])) # 0 or 1
    iterations = int(sys.argv[4]) # number
    model_names = sys.argv[5:] # space separated list

    for model_name in model_names:
        print(f"Testing {model_name} model.")
        if test_type == "all":
            run_tests("base", save_to_file, iterations, show_visual, model_name)
            run_tests("superquadric", save_to_file, iterations, show_visual, model_name)
        else:
            run_tests(test_type, save_to_file, iterations, show_visual, model_name)

    if (save_to_file):
        summarize_results()
