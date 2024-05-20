from model.geometry import *
import os
import torch
from tqdm import tqdm
import argparse
# from data._4DMatch import _4DMatch
from correspondence.datasets._4dmatch import _4DMatch
from model.registration import Registration
import yaml
from easydict import EasyDict as edict
from model.loss import compute_flow_metrics
from utils.benchmark_utils import setup_seed
from utils.utils import Logger, AverageMeter
from utils.tiktok import Timers
# from ma_dataset import MixamoAMASS
from df_dataset import DfaustTrain
from time import perf_counter

def eval_metric(pred, gt):
    # F, N, 3
    ate = torch.abs(pred-gt).mean()
    l2 = torch.norm(pred-gt,dim=-1)

    a01 = l2 < 0.01  # F, N
    d01 = torch.sum(a01).float() / (a01.shape[0] * a01.shape[1])

    a02 = l2 < 0.05  # F, N
    d02 = torch.sum(a02).float() / (a02.shape[0] * a02.shape[1])

    return ate, d01, d02


def to_cuda_float(batch, half=False):
    if isinstance(batch, tuple) or isinstance(batch, list):
        for i in range(len(batch)):
            batch[i] = to_cuda_float(batch[i], half)
    elif isinstance(batch, dict):
        for k in batch.keys():
            batch[k] = to_cuda_float(batch[k], half)
    else:
        try:
            if not half:
                batch = torch.from_numpy(batch).float().cuda()
            else:
                batch = torch.from_numpy(batch).half().cuda()
        except:
            batch = batch
    return batch



def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)

setup_seed(0)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help= 'Path to the config file.')
    parser.add_argument('--visualize', action = 'store_true', help= 'visualize the registration results')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['folder'], config['exp_dir'])
    os.makedirs(config['snapshot_dir'], exist_ok=True)


    config = edict(config)


    # backup the experiment
    os.system(f'cp -r config {config.snapshot_dir}')
    os.system(f'cp -r data {config.snapshot_dir}')
    os.system(f'cp -r model {config.snapshot_dir}')
    os.system(f'cp -r utils {config.snapshot_dir}')


    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')


    model = Registration(config)
    timer = Timers()

    eval_dict = {'ate': [], '0.1': [], '0.2': [], 'time': []}

    # splits = ['4DMatch-F', '4DLoMatch-F']

    # for benchmark in splits:

    # config.split['test'] = benchmark

    D = DfaustTrain(root_dir='data/DFAUST') 
    # logger = Logger(  os.path.join( config.snapshot_dir, benchmark+".log" ))

    stats_meter = None

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
        # p0 = torch.stack(traj_pred).cpu().numpy()
        # p1 = batch['tracks'].cpu().numpy()
        # p = np.concatenate([p0,p1],axis=1)
        # c0 = np.ones_like(p0[0]) * np.array([1.,0.,0.])
        # c1 = np.ones_like(p1[0]) * np.array([0.,1.,0.])
        # c = np.concatenate([c0,c1],axis=0)
        # from cvtb import vis
        # vis.pcd(p, c)
        # breakpoint()


    np.save('cndp_df.npy', eval_dict)

    # # note down flow scores on a benchmark
    # message = f'{i}/{len(D)}: '
    # for key, value in stats_meter.items():
    #     message += f'{key}: {value.avg:.3f}\t'
    # logger.write(message + '\n')
    # print( "score on ", benchmark, '\n', message)


    # # note down average time cost
    # print('time cost average')
    # for ele in timer.get_strings():
    #     logger.write(ele + '\n')
    #     print(ele)
