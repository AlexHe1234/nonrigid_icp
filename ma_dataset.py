import numpy as np
from torch.utils import data
from typing import Literal, List
from data_utils import *
import json
import os
import fpsample


class MixamoAMASS(data.Dataset):
    def __init__(self,
                 split: Literal['train', 'val', 'test'],  # passed in by main.py

                 root_dir: str,
                 num_tracks: int=256,
                 num_points: int=2048,
                 max_frame: int=50,

                 rand_rotation: bool=True,
                 rand_rotate_z_only: bool=True,
                 rand_padding: bool=True,
                 rand_padding_range: List[float]=[0.05, 0.1],
                 rand_reordering: bool=False,
                 rand_perturb: bool=True,
                 rand_perturb_avg_sd: List[float]=[0., 1.e-5],

                 uni_prob: float=1.,
                 noise_ratio: float=0.001,
                 fps_method: str='fpsample',

                 **kwargs,
                 ):
        super().__init__()
        assert split in ['train', 'val', 'test'], f'Split type {split} is not supported'
        self.split = split
        if self.split != 'train':
            np.random.seed(42)

        self.num_tracks = num_tracks
        self.num_points = num_points
        self.max_frame = max_frame

        self.rand_rotation = rand_rotation
        self.rand_rotate_z_only = rand_rotate_z_only
        self.rand_padding = rand_padding
        self.rand_padding_range = rand_padding_range
        self.rand_reordering = rand_reordering
        self.rand_perturb = rand_perturb
        self.rand_perturb_avg_sd = rand_perturb_avg_sd
        self.uni_prob = uni_prob
        self.noise_ratio = noise_ratio
        self.fps_method = fps_method

        self.root_dir = root_dir
        if self.split == 'train':
            sf = os.path.join(self.root_dir, '0train.json')
        elif self.split == 'val':
            sf = os.path.join(self.root_dir, '0val.json')
        else:
            sf = os.path.join(self.root_dir, '0test.json')
            
        with open(sf, 'r') as f:
            seqs = json.load(f)
        
        self.seqs = seqs
        # seqs = sorted([f for f in os.listdir(self.root_dir) if f[-3:] == 'npy'])
        # if self.split == 'train':
        #     self.seqs = seqs[2:]
        # elif self.split == 'val':
        #     self.seqs = seqs[:2]
        # else:
        #     raise ValueError(f'Split type {self.split} is not supported')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        seq_ = np.load(os.path.join(self.root_dir, seq_name))  # F_og, P_og, 3

        nf = seq_.shape[0]
        if nf > self.max_frame:
            start_frame = np.random.randint(0, nf - self.max_frame + 1)
            seq = seq_[start_frame : start_frame + self.max_frame]
            nf = seq.shape[0]

        # gt_seq_copy = seq.copy()

        # if self.track_mesh:
        rand_mesh_id = np.random.randint(0, len(seq_))
        gt_verts = np.concatenate([seq_[rand_mesh_id][None], seq], axis=0)
            
        # nf = gt_verts.shape[0]
        # gt_verts_copy = gt_verts.copy()

        gt_verts, reverse_transform = proc_frame_points(gt_verts, 
                                                        ret_reverse_transform=True,
                                                        use_rand_rotation=False,
                                                        rand_rotate_z_only=self.rand_rotate_z_only,
                                                        use_rand_padding=False,
                                                        rand_padding_range=self.rand_padding_range)
        
        transform = np.eye(4)[None]
        for tran_mat in reversed(reverse_transform):
            transform = np.matmul(transform, tran_mat)  # this is transposed!

        transform_ = transform[1:]

        feature_weight_gt = np.ones((transform_.shape[0], self.num_tracks))
        first_appear = np.zeros((self.num_tracks))
        
        # random sample points
        aug_verts = gt_verts[1:].copy()
        sample_num_points = min(self.num_points, gt_verts.shape[1])
        if self.rand_perturb:
            aug_verts += np.random.randn(*aug_verts.shape) * self.rand_perturb_avg_sd[1] + self.rand_perturb_avg_sd[0]    
        rand_sample_buffer = np.empty((nf, sample_num_points, 3))  # resample mesh, weighted by face area
        num_point_samples = int(rand_sample_buffer.shape[1] * (1. - self.noise_ratio))
        # num_over_samples = int(self.over_sampling * num_point_samples)
        for i in range(nf): 
            rand_sample_buffer_i = fps(aug_verts[i], num_point_samples, method=self.fps_method)
            rand_sample_buffer[i] = noisify_pcd(rand_sample_buffer_i, rand_sample_buffer.shape[1] - num_point_samples)

        ret = {
            'reverse_transform': transform_,  # nf, 4, 4 c 
            # 'reverse_transform_mesh': transform[0],  # 4, 4 c
            'tracks': gt_verts[1:],  # nf, ntrack, 3 mesh gt
            # 'tracks_og': gt_verts_copy[:, sampled_verts_indices],  # nf, ntrack, 3
            'points': rand_sample_buffer,  # nf, npoint, 3  sparse sampled points
            'first': first_appear,  # ntrack,
            'f_weights': feature_weight_gt,  # nf, ntrack
        }

        ret['points_mesh'] = gt_verts[0] # N, 3
        mesh_anchor_indices = fpsample.bucket_fps_kdline_sampling(gt_verts[0], self.num_tracks, 5)
        ret['tracks_mesh'] = gt_verts[0][mesh_anchor_indices]
        # ret['tracks_mesh_gt'] = gt_verts[1:][:, mesh_anchor_indices]

        return ret
