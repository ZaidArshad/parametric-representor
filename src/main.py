from optimization import *
from utilities import *
from superquadric import *
from segmentation import *
from random import randint
import trimesh

if __name__ == "__main__":

    # load point cloud
    pc = trimesh.load("data/hand/surface_points.ply")
    points = pc.vertices

    # initialize the arguments to the fitting algorithm
    args = {'inlier_ratio': 0.999}
    min_cluster_size = 5
    superquadrics = []
    clusters = [points]

    # recursively fit superquadrics to the point cloud
    i = 0
    while i < len(clusters):

        # fit a superquadric to the current cluster of points
        x, (outliers, inliers) = points_to_superquadric(clusters[i], args=args)
        superquadrics.append(Superquadric(x))

        # update the current cluster
        clusters[i] = inliers

        # add the clusters of outliers for subsequent iterations
        if len(outliers) > 2:
            clusters += cluster_points(outliers, min_cluster_size)

        i += 1

    # visualize fitted superquadrics
    colors = [[randint(0, 255), randint(0, 255), randint(0, 255), 127] for _ in clusters]
    sq_meshes = [superquadrics[i].create_mesh(u_res=20, v_res=20, colors=colors[i]) for i in range(len(superquadrics))]
    pc_meshes = [trimesh.points.PointCloud(clusters[i], colors=colors[i]) for i in range(len(clusters))]

    scene = trimesh.Scene([*sq_meshes, *pc_meshes])
    scene.show(background=[0, 0, 0, 255])
