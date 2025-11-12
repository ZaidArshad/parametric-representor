import trimesh
import numpy as np

e1 = 1.7
e2 = 1
a = [1, 1, 1]
points = []

def cos_pow(x, p):
    return np.sign(np.cos(x)) * np.abs(np.cos(x))**p

def sin_pow(x, p):
    return np.sign(np.sin(x)) * np.abs(np.sin(x))**p

for n in np.arange(-np.pi/2, np.pi/2, 0.1):
    for w in np.arange(-np.pi, np.pi, 0.2):
        x = a[0]*cos_pow(n,e1)*cos_pow(w,e2)
        y = a[1]*cos_pow(n,e1)*sin_pow(w,e2)
        z = a[2]*sin_pow(n,e1)
        point = [x, y, z]
        points.append(point)

np_points = np.array(points)
superquadric = trimesh.points.PointCloud(
    vertices=np_points, colors=[0, 0, 255, 255]
)

scene = trimesh.Scene([superquadric])
scene.show()