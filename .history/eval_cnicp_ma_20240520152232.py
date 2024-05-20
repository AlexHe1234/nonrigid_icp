import numpy as np
import open3d as o3d
import copy

from icp import icp,draw_registration_result
from nricp import nonrigidIcp

from ma_dataset import MixamoAMASS
import torch
from time import perf_counter
from tqdm import tqdm


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


def reg(sourcemesh, targetmesh):  # numpy arrays
    initial_guess = np.eye(4)
    affine_transform = icp(sourcemesh,targetmesh,initial_guess)

    refined_sourcemesh = copy.deepcopy(sourcemesh)
    refined_sourcemesh.transform(affine_transform)
    refined_sourcemesh.compute_vertex_normals()

    deformed_mesh = nonrigidIcp(refined_sourcemesh,targetmesh)

    return np.asarray(deformed_mesh.vertices)


D = MixamoAMASS(split='test',root_dir='data/mixamo_cmu')
eval_dict = {'ate': [], '0.1': [], '0.2': [], 'time': []}


for i in tqdm(range( len(D))):
    # src_pcd, tar_pcd, flow_gt = dat

    batch = D.__getitem__(i)
    batch = to_cuda_float(batch)
    frame = len(batch['points'])
    
    traj_pred = []
    
    # timer.tic("registration")
    start_time = perf_counter()

    for f in (range(frame)):
        if f == 0:
            src_pcd = batch['points_mesh']
            # breakpoint()
            tgt_pcd = batch['points'][0]
            # flow_gt = batch['tracks'][0] - batch['points_mesh']
        else:
            src_pcd = warped_pcd.detach()
            tgt_pcd = batch['points'][f]
            # flow_gt = batch['tracks'][1] - 


        """obtain overlap mask"""
        overlap = np.ones(len(src_pcd))

        model.load_pcds(src_pcd, tgt_pcd)

        warped_pcd, iter_cnt, timer = model.register(visualize=args.visualize, timer = timer)
        # flow = warped_pcd - model.src_pcd
        # metric_info = compute_flow_metrics(flow, flow_gt, overlap=overlap)
        traj_pred.append(warped_pcd.detach().clone())

        # if stats_meter is None:
        #     stats_meter = dict()
        #     for key, _ in metric_info.items():
        #         stats_meter[key] = AverageMeter()
        # for key, value in metric_info.items():
            # stats_meter[key].update(value)
    # timer.toc("registration")
    end_time = perf_counter()
    traj_pred = torch.stack(traj_pred)
    ate, d01, d02 = eval_metric(traj_pred, batch['tracks'])

    # breakpoint()

    eval_dict['ate'].append(ate)
    eval_dict['0.1'].append(d01)
    eval_dict['0.2'].append(d02)
    t = (end_time - start_time) / traj_pred.shape[0]
    eval_dict['time'].append(t)

    print(f'ate {ate} d01 {d01} d02 {d02} time {t}')

np.save('cndp_df.npy', eval_dict)


