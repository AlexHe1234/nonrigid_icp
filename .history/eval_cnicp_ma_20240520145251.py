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


def reg(src_pcd, tar_pcd):  # numpy arrays
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

