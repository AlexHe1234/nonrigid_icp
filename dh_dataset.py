import numpy as np
from typing import Literal, List
import os
from os.path import join as osp
from torch.utils import data
from data_utils import *
import json
# for testing


# specific for training
class DeformingThings(data.Dataset):
    def __init__(self, 
                 split: Literal['train', 'val', 'test'],  # passed in by main.py
                 
                 root_dir: str,

                 
                 **kwargs,
                 ) -> None:
        
        split = 'test'  # passed in by main.py
                 
        # root_dir: str
        num_tracks: int = 256
        num_points: int = 2048
        num_mesh_points: int = 100000
        max_frame: int = 50
        
        rand_start: bool = True
        rand_start_min_length: int= 48
        rand_scaling: bool = False
        rand_scaling_range: List[float]=[0.5, 1.5]
        rand_rotation: bool=True
        rand_rotate_z_only: bool=True
        rand_padding: bool=True
        rand_padding_range: List[float]=[0.05, 0.1]
        rand_reordering: bool=False
        rand_perturb: bool=True
        rand_perturb_avg_sd: List[float]=[0., 1.e-5]

        uni_prob: float=1.
        noise_ratio: float=0.00
        over_sampling: float=3.
        fps_method: str='fpsample'

        depth_proj_prob: float=0.
        track_mesh: bool=True
        rand_query_mesh_prob: float=True
        
        knn: int=4
        
        
        
        super().__init__()
        assert split in ['train', 'val', 'test'], f'Split type {split} is not supported'
        self.split = split
        if self.split != 'train':
            np.random.seed(42)
        
        self.root_dir = root_dir
        
        self.max_frame = max_frame
        self.num_tracks = num_tracks
        self.num_points = num_points
        self.num_mesh_points = num_mesh_points
        self.knn = knn
        
        self.rand_start = rand_start
        self.rand_start_min_length = int(rand_start_min_length)
        
        self.rand_scaling = rand_scaling
        self.rand_scaling_range = rand_scaling_range
        
        self.rand_rotation = rand_rotation
        self.rand_rotate_z_only = rand_rotate_z_only
        
        self.rand_padding = rand_padding
        self.rand_padding_range = rand_padding_range
        
        self.rand_reordering = rand_reordering
        
        self.rand_perturb = rand_perturb
        self.rand_perturb_avg_sd = rand_perturb_avg_sd

        self.uni_prob = uni_prob
        self.noise_ratio = noise_ratio
        self.over_sampling = over_sampling
        self.fps_method = fps_method

        self.depth_proj_prob = depth_proj_prob
        self.track_mesh = track_mesh
        self.rand_mesh = rand_query_mesh_prob

        if self.split == 'train':
            split_file0 = 'train_animals.json'
            split_file1 = 'train_humanoids.json'
        elif self.split == 'val':
            split_file0 = 'val_animals.json'
            split_file1 = 'val_humanoids.json'
        else:
            split_file0 = 'test_humanoids.json'
            split_file1 = None
        
        self.animes = []

        with open(os.path.join(self.root_dir, split_file0)) as f:
            split_data0 = json.load(f)  # list
        # search through root directory
        for name in split_data0:
            self.animes.append(osp(self.root_dir, name))

        if split_file1 is not None:
            with open(os.path.join(self.root_dir, split_file1)) as f:
                split_data1 = json.load(f)  # list
            # search through root directory
            for name in split_data1:
                self.animes.append(osp(self.root_dir, name))
            
    def __len__(self):
        return len(self.animes)

    def __getitem__(self, idx):
        reverse_transform = []
        nf, nv, nt, vert_data, face_data, offset_data = anime_read(self.animes[idx])  # int, int, int, Nv*3, Nt*3, (Nf-1)*Nv*3
        
        assert not np.isnan(vert_data).any()
        assert not np.isnan(face_data).any()
        assert not np.isnan(offset_data).any()
        assert nf == offset_data.shape[0] + 1

        gt_verts_all = vert_data + offset_data  # nf, nv, 3
        gt_verts_all = np.concatenate([vert_data[None], gt_verts_all], axis=0)  # F, N, 3

        # randomly choose frames
        if nf > self.max_frame:
            start_frame = np.random.randint(0, nf - self.max_frame + 1)
            gt_verts = gt_verts_all[start_frame : start_frame + self.max_frame].copy()
        else:
            start_frame = 0
            gt_verts = gt_verts_all.copy()

        if self.track_mesh:
            rand_mesh = np.random.uniform() < self.rand_mesh
            if rand_mesh:
                chosen_id = np.random.randint(nf)
            else:
                chosen_id = start_frame
            query_mesh = gt_verts_all[chosen_id]
            gt_verts = np.concatenate([query_mesh[None], gt_verts], axis=0)
            
        nf = gt_verts.shape[0]
        gt_verts_copy = gt_verts.copy()

        gt_verts, reverse_transform = proc_frame_points(gt_verts, 
                                                        ret_reverse_transform=True,
                                                        use_rand_rotation=self.rand_rotation,
                                                        rand_rotate_z_only=self.rand_rotate_z_only,
                                                        use_rand_padding=self.rand_padding,
                                                        rand_padding_range=self.rand_padding_range)
        
        # pcd seq aug, nothing to do with gt
        aug_verts = gt_verts.copy()

        depth_proj = np.random.uniform() < self.depth_proj_prob

        if depth_proj:
            # project first
            subsampled_verts = []
            dpts = []
            SIZE = 300
            k, r, t = generate_random_semisphere_camera(radius=2, fov=60, size=SIZE)
            sample_num_points = self.num_points
            for f in range(nf):
                subsampled_vert, dpt = back_project_depth(aug_verts[f], face_data, SIZE, k, r, t, return_dpt=True)
                dpts.append(dpt)
                if sample_num_points > subsampled_vert.shape[0]:
                    sample_num_points = subsampled_vert.shape[0]
                subsampled_verts.append(subsampled_vert)
            dpts = np.stack(dpts)  # nf, SIZE, SIZE

            rand_sample_buffer = np.empty((nf, sample_num_points, 3)) 

            num_point_samples = int(rand_sample_buffer.shape[1] * (1. - self.noise_ratio))
            for i in range(nf): 
                rand_sample_buffer_i = fps(subsampled_verts[i], num_point_samples, method=self.fps_method)
                rand_sample_buffer[i] = noisify_pcd(rand_sample_buffer_i, rand_sample_buffer.shape[1] - num_point_samples)

            if self.rand_reordering:  # reorder
                rand_sample_buffer = rand_reorder_by_frame(rand_sample_buffer)
                
            if self.rand_perturb:
                rand_sample_buffer += np.random.randn(*rand_sample_buffer.shape) * self.rand_perturb_avg_sd[1] + self.rand_perturb_avg_sd[0]    
        else:
            if self.split == 'train':
                sample_num_points = self.num_points
            else:
                sample_num_points = min(self.num_points, nv)

            if self.rand_reordering:  # reorder
                aug_verts = rand_reorder_by_frame(aug_verts)
                
            if self.rand_perturb:
                aug_verts += np.random.randn(*aug_verts.shape) * self.rand_perturb_avg_sd[1] + self.rand_perturb_avg_sd[0]    

            # vis_pcd(aug_verts[0])
            rand_sample_buffer = np.empty((nf, sample_num_points, 3))  # resample mesh, weighted by face area

            num_point_samples = int(rand_sample_buffer.shape[1] * (1. - self.noise_ratio))
            num_over_samples = int(self.over_sampling * num_point_samples)
            for i in range(nf): 
                rand_sample_buffer_i_over = rand_uniform_sampler(aug_verts[i], face_data, num_over_samples)
                rand_sample_buffer_i = fps(rand_sample_buffer_i_over, num_point_samples, method=self.fps_method)
                rand_sample_buffer[i] = noisify_pcd(rand_sample_buffer_i, rand_sample_buffer.shape[1] - num_point_samples)

        rand_sample_buffer[rand_sample_buffer > 1.] = 1. - 1e-5
        rand_sample_buffer[rand_sample_buffer < -1.] = -1. + 1e-5
        
        # sampled_verts_indices = np.random.choice(nv, size=self.num_tracks, replace=False)
        sampled_verts_indices = smart_sample(gt_verts[0], self.num_mesh_points, self.uni_prob)
        sampled_verts = gt_verts[:, sampled_verts_indices]
        
        transform = np.eye(4)[None]
        for tran_mat in reversed(reverse_transform):
            transform = np.matmul(transform, tran_mat)

        if self.track_mesh:
            points_mesh = sampled_verts[0]
            tracks_mesh = sampled_verts[0]
            transform_ = transform[1:]
            sampled_verts = sampled_verts[1:]
            gt_verts_copy = gt_verts_copy[1:]
            rand_sample_buffer = rand_sample_buffer[1:]

        feature_weight_gt = np.ones((sampled_verts.shape[0], self.num_tracks))
        first_appear = np.zeros((self.num_tracks))

        assert rand_sample_buffer.shape[0] > 0

        ret = {
            'reverse_transform': transform_,  # nf, 4, 4
            'reverse_transform_mesh': transform[0],  # 4, 4
            'tracks': sampled_verts,  # nf, ntrack, 3
            'tracks_og': gt_verts_copy[:, sampled_verts_indices],  # nf, ntrack, 3
            'points': rand_sample_buffer,  # nf, npoint, 3
            'first': first_appear,  # ntrack,
            'f_weights': feature_weight_gt,  # nf, ntrack
        }

        ret['points_mesh'] = points_mesh  # N, 3
        mesh_anchor_indices = fpsample.bucket_fps_kdline_sampling(points_mesh, self.num_tracks, 5)
        ret['tracks_mesh'] = points_mesh[mesh_anchor_indices]
        ret['tracks_mesh_gt'] = sampled_verts[:, mesh_anchor_indices]

        # np.save('doggieMN5_runforwardRM_mesh.npy', face_data)
        # exit()

        # point_traj = gt_verts_all[:, sampled_verts_indices]  # Fa, N, 3
        # anchor_traj = point_traj[:, mesh_anchor_indices]  # Fa, T, 3
        # dist = np.linalg.norm(point_traj[:, :, None] - anchor_traj[:, None, :], axis=-1).transpose((1, 2, 0))  # N, T, Fa
        # d_dist = dist.max(axis=-1) - dist.min(axis=-1)  # N, T
        # sim = np.exp(-d_dist)  # N, T
        # sim = (sim - sim.min()) / (sim.max() - sim.min())
        # ret['sim'] = sim ** 100
        
        # s = {'w': sim, 'p': points_mesh, 'a': ret['tracks_mesh']}
        # np.save('res.npy', s)
        # exit()
            
        # ret['knn'] = knn(ret['tracks_mesh'], self.knn)  # T, K
        # points_mesh should match tracks

        return ret
