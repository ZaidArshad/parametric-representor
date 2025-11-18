import trimesh
import numpy as np

e1 = 2.3
e2 = 1.2
a = [1, 1, 1]

resolution = 32

def main():

    vertices = []
    faces = []

    for n in np.linspace(-np.pi / 2, np.pi / 2, resolution // 2):
        for w in np.linspace(-np.pi, np.pi, resolution + 1):

            x = a[0] * cos_pow(n, e1) * cos_pow(w, e2)
            y = a[1] * cos_pow(n, e1) * sin_pow(w, e2)
            z = a[2] * sin_pow(n, e1)

            vertices.append([x, y, z])

    for n in range(resolution // 2 - 1):
        for w in range(resolution):

            faces.append([w + n * (1 + resolution), (1 + w) + n * (1 + resolution), w + (1 + n) * (1 + resolution)])
            faces.append([w + (1 + n) * (1 + resolution), (1 + w) + n * (1 + resolution), (1 + w) + (1 + n) * (1 + resolution)])

    superquadric = trimesh.Trimesh(vertices, faces)

    scene = trimesh.Scene([superquadric])
    scene.show()

def cos_pow(x, p):
    return np.sign(np.cos(x)) * np.pow(np.abs(np.cos(x)), p)

def sin_pow(x, p):
    return np.sign(np.sin(x)) * np.pow(np.abs(np.sin(x)), p)

if __name__ == '__main__': main()
