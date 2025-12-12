import numpy as np
import trimesh
from utilities import euler_to_matrix, matrix_to_euler

class Superquadric:
    """ class for defining and visualizing superquadrics"""
    def __init__(self, x):
        """ x = [e1, e2, a1, a2, a3, rx, ry, rz, tx, ty, tz] """

        self.x = x
        
        # shape
        self.e1 = x[0]
        self.e2 = x[1]
        self.shape = (self.e1, self.e2)

        # size
        self.a1 = x[2]
        self.a2 = x[3]
        self.a3 = x[4]
        self.size = (self.a1, self.a2, self.a3)

        # rotation
        self.rotation = (x[5], x[6], x[7])

        rx, ry, rz = self.rotation
        R3 = euler_to_matrix(rx, ry, rz)
        T = np.eye(4)
        T[:3, :3] = R3
        self.rotation_matrix = T

        # position
        self.position = (x[8], x[9], x[10])

    def get_similars(self):
        """ returns other superquadrics with a similar shape """

        similars = []

        # get similar superquadrics by axis-mismatch similarity
        rotation_matrix_1 = self.rotation_matrix[:3, (1, 2, 0)]
        rotation_matrix_2 = self.rotation_matrix[:3, (2, 0, 1)]

        similars.append(Superquadric(np.array([
            self.e2, self.e1,
            self.a2, self.a3, self.a1,
            *matrix_to_euler(rotation_matrix_1),
            *self.position,
            ])))

        similars.append(Superquadric(np.array([
            self.e2, self.e1,
            self.a3, self.a1, self.a2,
            *matrix_to_euler(rotation_matrix_2),
            *self.position,
            ])))

        # find duals for the current superquadric and its other similars
        for sq in (self, *similars):

            # get a similar superquadric by duality similarity
            duality_similarity = np.abs(sq.a1 / sq.a2)

            if 0.8 < duality_similarity and duality_similarity < 1.2:

                # compute scaling factor s
                if sq.e2 <= 1:
                    s = (1 - np.sqrt(2)) * sq.e2 + np.sqrt(2)
                else:
                    s = (np.sqrt(2) / 2 - 1) * sq.e2 + 2 - np.sqrt(2) / 2

                # compute average axis scale
                a = s * (sq.a1 + sq.a2) / 2

                # compute the new rotation matrix
                rotation_matrix = sq.rotation_matrix[:3, :3] @ euler_to_matrix(0, 0, np.pi / 4)

                similars.append(Superquadric(np.array([
                    sq.e1, 2 - sq.e2,
                    a, a, sq.a3,
                    *matrix_to_euler(rotation_matrix),
                    *sq.position,
                    ])))

        return similars

    def get_splits(self):
        """ returns two superquadrics approximating the current superquadric and the origin and normal vector of the plane between them """

        splits = []

        normal = (self.rotation_matrix @ [0, 0, 1, 0])[:3]

        if self.e1 < 0.5:

            splits.append(Superquadric(np.array([
                self.e1, self.e2,
                self.a1, self.a2, self.a3 / 2,
                *self.rotation,
                *(self.position + normal * self.a3 / 2),
                ])))

            splits.append(Superquadric(np.array([
                self.e1, self.e2,
                self.a1, self.a2, self.a3 / 2,
                *self.rotation,
                *(self.position - normal * self.a3 / 2),
                ])))

        return splits, self.position, normal

    def create_mesh(self, u_res=50, v_res=50, colors=[150, 150, 250, 127]):
        """ creates a mesh of the superquadric surface """
        
        # parameter ranges
        u = np.linspace(0, 2 * np.pi, u_res, endpoint=False)
        v = np.linspace(0, np.pi, v_res, endpoint=True)
        u, v = np.meshgrid(u, v, indexing='ij')
        u, v = u.ravel(), v.ravel()

        # signed power
        def spow(x, p):
            return np.sign(x) * np.pow(np.abs(x), p)

        # parametric superquadric surface
        points = np.zeros((u_res * v_res, 3))
        points[:, 0] = spow(np.sin(v), self.e1) * spow(np.cos(u), self.e2)
        points[:, 1] = spow(np.sin(v), self.e1) * spow(np.sin(u), self.e2)
        points[:, 2] = spow(np.cos(v), self.e1)

        # scaling
        points *= self.size

        # rotation + translation
        points = points @ self.rotation_matrix[:3, :3].T + self.position

        # build faces for the mesh
        faces = []
        for i in range(u_res):
            for j in range(v_res - 1):
                idx00 = ( i      % u_res) * v_res + ( j      % v_res)
                idx01 = ( i      % u_res) * v_res + ((j + 1) % v_res)
                idx10 = ((i + 1) % u_res) * v_res + ( j      % v_res)
                idx11 = ((i + 1) % u_res) * v_res + ((j + 1) % v_res)

                faces.append([idx00, idx01, idx10])
                faces.append([idx01, idx11, idx10])

        faces = np.array(faces)

        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=True)

        # change color
        mesh.visual.vertex_colors = colors

        return mesh

    def generate_points(self, num_points=2000):
        """ uniformly samples points on the superquadric mesh, used for initial fitting tests """
        
        # create mesh and sample points
        mesh = self.create_mesh(u_res=int(np.sqrt(num_points)), v_res=int(np.sqrt(num_points)))
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
