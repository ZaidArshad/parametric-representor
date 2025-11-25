# This file contains function related to our evaluation metrics

import numpy as np
from scipy.spatial import KDTree
import trimesh

# From https://medium.com/@sim30217/chamfer-distance-4207955e8612
def chamfer_distance(a, b):
    """ Calculates the chamfer distance between 2 sets of points.

    Args: 
        A (list): List of points
        B (list): List of points

    Returns:
        float: Chamfer distance between the set of points.
    """
    tree = KDTree(b)
    dist_A = tree.query(a)[0]
    tree = KDTree(a)
    dist_B = tree.query(b)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def reverse_flood_fill(voxels):
    """
    Starts with a grid of voxels that disintegrates until it touches
    the input voxel. This create a filled representation of the filled
    voxel. 
    
    Args:
        voxels (set): Set of points
    
    Returns:
        Original set of point + filling points
    """
    points = np.array(list(voxels)) # might not need to cast to list
    max_point = np.max(points, axis=0) + 1
    min_point = np.min(points, axis=0) - 1
    
    unseen_points = [min_point]
    grid_voxels = {
        (x, y, z)
        for x in range(min_point[0], max_point[0])
        for y in range(min_point[1], max_point[1])
        for z in range(min_point[2], max_point[2])
    }

    while unseen_points:
       (x, y, z) = unseen_points.pop()
       current = (x, y, z)

       if current in voxels or current not in grid_voxels:
           continue
       
       grid_voxels.remove(current)
       neighbours = [
           (x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)
       ]
       unseen_points.extend(neighbours)

    return grid_voxels

def pc_to_voxel(pc, merge_len=0.1):
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
    voxels = reverse_flood_fill(voxels)
    return voxels

def approx_pc_surface_area(pc):
    """ Approximating the area of a point cloud using voxels.
    
    Args:
        pc (list): Set of points in point cloud

    Returns:
        int: Number of voxels that approximate point cloud. 
    """
    return len(pc_to_voxel(pc))

def voxel_to_mesh(voxels, voxel_size=1.0):
    """ Creating a cube mesh representation of voxels.
    
    Args:
        pc (list): List of points in point cloud
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

def pc_intersection_over_union(a, b):
    """
    Gets the intersection over union ratio of 2 point clouds
    
    Args:
        a (list): List of points in point cloud
        b (list): List of points in point cloud
    
    Returns:
        float: Intersection over union ratio
    """
    a_vx = pc_to_voxel(a)
    b_vx = pc_to_voxel(b)
    union = a_vx.union(b_vx)
    intersection = a_vx.intersection(b_vx)
    return len(intersection)/len(union)
    
# Test
# dog_pc = trimesh.load("data/dog/surface_points.ply")
# dog_pc_2 = dog_pc.copy()
# offset = (0.1,0.1,0.1)
# dog_pc_2.apply_translation(offset)
# print("IOU", pc_intersection_over_union(dog_pc, dog_pc_2))

# a = pc_to_voxel(dog_pc)
# mesh_a = voxel_to_mesh(a, 0.9)
# mesh_a.visual.face_colors = [[255, 0, 0, 255]] * len(mesh_a.faces)
# b = pc_to_voxel(dog_pc_2)
# mesh_b = voxel_to_mesh(b, 0.9)
# mesh_b.visual.face_colors = [[0, 0, 255, 255]] * len(mesh_b.faces)
# scene = trimesh.Scene([mesh_a, mesh_b])
# scene.show()




