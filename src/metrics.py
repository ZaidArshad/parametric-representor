# This file contains function related to our evaluation metrics

import numpy as np
from scipy.spatial import KDTree
import trimesh

# From https://medium.com/@sim30217/chamfer-distance-4207955e8612
def chamfer_distance(A, B):
    """ Calculates the chamfer distance between 2 sets of points.

    Args: 
        A (list): Set of points
        B (list): Set of points

    Returns:
        float: Chamfer distance between the set of points.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def point_cloud_to_voxel(pc, merge_len=0.1):
    """ Converts a point cloud to voxel space. 
    Uses sets to allow for set theory.
    
    Args:
        pc (list): Set of points in point cloud
        merge_len (float, optional): Range to merge points

    Returns:
        set: The point cloud in voxel space.
    """
    voxels = set()
    for point in pc:
        voxels.add((
            int(point[0] / merge_len),
            int(point[1] / merge_len),
            int(point[2] / merge_len),
        ))
    return voxels

def point_cloud_as_voxel_area(pc):
    """ Approximating the area of a point cloud using voxels.
    
    Args:
        pc (list): Set of points in point cloud

    Returns:
        int: Number of voxels that approximate point cloud. 
    """
    return len(point_cloud_to_voxel(pc))

def voxel_to_mesh(voxels, voxel_size=1.0):
    """ Creating a cube mesh representation of voxels.
    
    Args:
        pc (list): Set of points in point cloud
        voxel_size (float, optional): Cube extent of each voxel

    Returns:
        Trimesh: Combined cube mesh for each voxel. 
    """
    cubes = []
    for vox in voxels:
        cube = trimesh.creation.box(
            transform=trimesh.transformations.translation_matrix(vox),
            extents=[voxel_size,voxel_size,voxel_size]
        )
        cubes.append(cube)
    return trimesh.util.concatenate(cubes)

""" Example 

dog_pc = trimesh.load("data/dog/surface_points.ply")
dog_vx = list(point_cloud_to_voxel(dog_pc))
voxel_to_mesh(dog_vx, 0.9).show()
"""