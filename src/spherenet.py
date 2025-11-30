# Retrofitted Task 3 from in class assignment.
# Part of the code in adopted from DualSDF repository.
import torch
import numpy as np
import torch.nn as nn
import trimesh
from dgcnn import DGCNNFeat

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix


def determine_sphere_sdf(query_points, sphere_params):
    """Query sphere sdf for a set of points.

    Args:
        query_points (torch.tensor): Nx3 tensor of query points.
        sphere_params (torch.tensor): Kx4 tensor of sphere parameters (center and radius).

    Returns:
        torch.tensor: Signed distance field of each sphere primitive with respect to each query point. NxK tensor.
    """

    sphere_sdf = torch.linalg.norm(query_points.unsqueeze(1) - sphere_params[:, :3], dim=2) - sphere_params[:, 3]
    return sphere_sdf


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_ch = 256
        out_ch = 1024
        feat_ch = 512

        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, out_ch),
        )
        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        return out2


class SphereNet(nn.Module):
    def __init__(self, num_spheres=256):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

    def forward(self, surface_points, query_points):
        features = self.encoder(surface_points)
        sphere_params = self.decoder(features)
        
        # Decode features from 4Nx1 to Nx4 tensor and adjust our resolution between 0 and 1
        sphere_params = torch.sigmoid(sphere_params.view(-1, 4))
        # Translate spheres to proper position 
        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1]).to(sphere_params.device)
        # Scale spheres to size
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder

        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params


def visualise_spheres(sphere_params, reference_model):
    """
    Displays the sphere fitting against the reference model.

    Args:
        sphere_params (torch.tensor): Kx4 tensor of sphere parameters (center and radius).
        reference_model (list): List of points for a point cloud. 
    """
    sphere_params = sphere_params.cpu().detach().numpy()
    sphere_centers = sphere_params[..., :3]
    sphere_radii = np.abs(sphere_params[..., 3])
    scene = trimesh.Scene()
    if reference_model is not None:
        scene.add_geometry(trimesh.PointCloud(reference_model, colors=[255, 255, 0, 255]))
    for center, radius in zip(sphere_centers, sphere_radii): 
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2, colors=[150, 150, 250, 170])
        sphere.apply_translation(center)
        scene.add_geometry(sphere)
        
    scene.show(background=[0, 0, 0, 255])


def create_sphere_pc(sphere_param, res=10):
    """
    Creates a point cloud from sphere parameters.

    Args:
        sphere_param (torch.tensor): Sphere center and radius -> (x0, y0, z0, r).
        res (int, optional): How many sample to take from the sphere along an axis. 
    Returns:
        (ndarray): Nx3 matrix of points to make the point cloud 
    """

    # parameter ranges
    u = np.linspace(0, np.pi, res)
    v = np.linspace(0, 2*np.pi, res)
    u, v = np.meshgrid(u, v)
    r = sphere_param[3]

    # parametric sphere surface
    x = sphere_param[0] + r * np.sin(u)*np.cos(v)
    y = sphere_param[1] + r * np.sin(u)*np.sin(v)
    z = sphere_param[2] + r * np.cos(u)
    
    # stack into (N,3)
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    return points

def sphere_params_to_pc(sphere_params):
    """
    Generate a point cloud from a list of sphere params.
    
    Args:
        sphere_params (torch.tensor): Kx4 tensor of sphere parameters (center and radius).
    Returns:
        (list): Nx3 matrix of points to up a point cloud.
    """
    pc = []
    for sphere_param in sphere_params:
        pc.extend(create_sphere_pc(sphere_param))
    
    return pc

def determine_sphere_params(surface_points, sdf_points, sdf_values, num_spheres=256, num_epochs=100):
    """
    Using SphereNet to fit spheres to a given sdf model. 

    Args:
        surface_points (list): Nx3 matrix of surface points of the ground truth.
        sdf_points (list): Kx3 matrix of sdf points of the ground truth.
        sdf_values (list): List of K signed distances of the ground truth.
        num_spheres (int, optional): L number of spheres to fit.
        num_epochs (int, optional): Number of iterations.
    Returns:
        (list): Lx4 matrix of spheres parameters (x0, y0, z0, r).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sdf_points = torch.from_numpy(sdf_points).float().to(device)
    sdf_values = torch.from_numpy(sdf_values).float().to(device)
    surface_pointcloud = torch.from_numpy(surface_points).float().to(device)

    model = SphereNet(num_spheres=num_spheres).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for i in range(num_epochs):
        optimizer.zero_grad()
        sphere_sdf, sphere_params = model(
            surface_pointcloud.unsqueeze(0).transpose(2, 1), sdf_points
        )
        sphere_sdf = bsmin(sphere_sdf, dim=-1).to(device)
        mseloss = torch.mean((sphere_sdf - sdf_values)**2)

        loss = mseloss
        loss.backward()
        optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}")

    return sphere_params


# Test 
# pcd_model = trimesh.load("data/dog/surface_points.ply").vertices
# sdf_model = np.load("data/dog/voxel_and_sdf.npz")
# sdf_points = sdf_model["sdf_points"]
# sdf_values = sdf_model["sdf_values"]

# sphere_params = determine_sphere_params(pcd_model, sdf_points, sdf_values, num_epochs=50)
# # visualise_spheres(sphere_params, reference_model=pcd_model.vertices)
# pc = sphere_params_to_pc(sphere_params.tolist())
# pc.show()
