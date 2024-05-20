import numpy as np
import open3d as o3d
import copy

from icp import icp
from nricp import nonrigidIcp

from ma_dataset import MixamoAMASS
import torch
from time import perf_counter
from tqdm import tqdm
from cvtb import vis


def eval_metric(pred, gt):
    # F, N, 3
    ate = torch.abs(pred-gt).mean()
    l2 = torch.norm(pred-gt,dim=-1)

    a01 = l2 < 0.01  # F, N
    d01 = torch.sum(a01).float() / (a01.shape[0] * a01.shape[1])

    a02 = l2 < 0.05  # F, N
    d02 = torch.sum(a02).float() / (a02.shape[0] * a02.shape[1])

    return ate, d01, d02


# s


def reg(sourcepcd, targetpcd):  # numpy arrays
    initial_guess = np.eye(4)
    affine_transform = icp(sourcepcd,targetpcd,initial_guess)
    
    sourceply =  o3d.geometry.PointCloud()
    sourceply.points = o3d.utility.Vector3dVector(sourcepcd)
    sourceply.transform(affine_transform)
    sourceply.estimate_normals()

    deformed_mesh = nonrigidIcp(sourceply,targetpcd)

    return deformed_mesh


D = MixamoAMASS(split='test',root_dir='data/mixamo_cmu')
eval_dict = {'ate': [], '0.1': [], '0.2': [], 'time': []}


for i in tqdm(range( len(D))):
    batch = D.__getitem__(i)
    frame = len(batch['points'])
    
    traj_pred = []
    
    start_time = perf_counter()

    for f in (range(frame)):
        if f == 0:
            src_pcd = batch['points_mesh']
            # breakpoint()
            tar_pcd = batch['points'][0]
            # flow_gt = batch['tracks'][0] - batch['points_mesh']
        else:
            src_pcd = warped_pcd
            tar_pcd = batch['points'][f]

        warped_pcd = reg(src_pcd, tar_pcd)

        traj_pred.append(warped_pcd.copy())
        breakpoint()

    end_time = perf_counter()
    traj_pred = torch.stack(traj_pred)
    ate, d01, d02 = eval_metric(traj_pred, batch['tracks'])


    eval_dict['ate'].append(ate)
    eval_dict['0.1'].append(d01)
    eval_dict['0.2'].append(d02)
    t = (end_time - start_time) / traj_pred.shape[0]
    eval_dict['time'].append(t)

    print(f'ate {ate} d01 {d01} d02 {d02} time {t}')

np.save('cndp_df.npy', eval_dict)


