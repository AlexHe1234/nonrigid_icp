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


def get_mesh(src_pcd):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(src_pcd)
    source_pcd.estimate_normals()
    src_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source_pcd)
    src_mesh.compute_vertex_normals()
    return src_mesh


def get_fake_mesh(tar_pcd):
    target_mesh = o3d.geometry.TriangleMesh()
    target_mesh.vertices = o3d.utility.Vector3dVector(tar_pcd)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(tar_pcd)
    target_pcd.estimate_normals()
    target_mesh.vertex_normals = target_pcd.normals
    return target_mesh


D = MixamoAMASS(split='test',root_dir='data/mixamo_cmu')


def reg(src_pcd, tar_pcd):  # numpy arrays

    targetmesh.compute_vertex_normals()

    initial_guess = np.eye(4)
    affine_transform = icp(sourcemesh,targetmesh,initial_guess)

    refined_sourcemesh = copy.deepcopy(sourcemesh)
    refined_sourcemesh.transform(affine_transform)
    refined_sourcemesh.compute_vertex_normals()

    deformed_mesh = nonrigidIcp(refined_sourcemesh,targetmesh)

