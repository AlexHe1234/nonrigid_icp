import numpy as np
import open3d as o3d
import copy

from icp import icp,draw_registration_result
from nricp import nonrigidIcp

from ma_dataset import MixamoAMASS
import torch



def eval_metric(pred, gt):
    # F, N, 3
    ate = torch.abs(pred-gt).mean()
    l2 = torch.norm(pred-gt,dim=-1)

    a01 = l2 < 0.01  # F, N
    d01 = torch.sum(a01).float() / (a01.shape[0] * a01.shape[1])

    a02 = l2 < 0.05  # F, N
    d02 = torch.sum(a02).float() / (a02.shape[0] * a02.shape[1])

    return ate, d01, d02


point_cloud = np.random.rand(1000, 3)  # Example random point cloud

# Convert numpy array to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Estimate normals
pcd.estimate_normals()

# Mesh generation
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Decimate mesh
mesh = mesh.simplify_quadric_decimation(10000)

# Smooth mesh
mesh = mesh.filter_smooth_laplacian(1)

# Visualize the result
o3d.visualization.draw_geometries([mesh])


def reg(src_pcd, tar_pcd):  # numpy arrays
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()

    source_pcd.points = o3d.utility.Vector3dVector(src_pcd)
    target_pcd.points = o3d.utility.Vector3dVector(tar_pcd)

    sourcemesh = o3d.io.read_triangle_mesh("data/source_test.obj")
    targetmesh = o3d.io.read_triangle_mesh("data/target_half.obj")
    sourcemesh.compute_vertex_normals()
    targetmesh.compute_vertex_normals()

    initial_guess = np.eye(4)
    affine_transform = icp(sourcemesh,targetmesh,initial_guess)

    refined_sourcemesh = copy.deepcopy(sourcemesh)
    refined_sourcemesh.transform(affine_transform)
    refined_sourcemesh.compute_vertex_normals()

    deformed_mesh = nonrigidIcp(refined_sourcemesh,targetmesh)

