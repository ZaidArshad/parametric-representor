from optimization import *
from utilities import *
from superquadric import *
import trimesh

if __name__ == "__main__":
    # load point cloud
    pc = trimesh.load("data/hand/surface_points.ply")
    points = pc.vertices

    # fit superquadric
    args = {
        'inlier_ratio': 0.999
        }

    x, (outliers, inliers) = points_to_superquadric(points, args=args)
    sq = Superquadric(x)

    # the outliers are the points not 'used' in the fitting for this superquadric
    # we can use these to fit additional superquadrics later (after segmentation)
    # we keep the inliers for visualization

    # visualize fitted superquadric and remaining points
    mesh = sq.create_mesh(u_res=100, v_res=100)
    pc_outliers_mesh = trimesh.points.PointCloud(outliers, colors=[255, 255, 0, 255])
    pc_inliers_mesh = trimesh.points.PointCloud(inliers, colors=[0, 0, 255, 255])
    scene = trimesh.Scene([mesh, pc_outliers_mesh, pc_inliers_mesh])
    scene.show(background=[0, 0, 0, 255])