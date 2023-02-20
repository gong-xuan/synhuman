import os, argparse
import torch

import configs
from model.model import Build_Model
from utils.io_utils import fetch_processed_pr_path, fetch_processed_img_path
from dataload.dataloader import Build_Test_Dataloader
from eval.test_hmr import testHMRImg, testHMRPr


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--batch_size', type=int, default=128) #125 for mpi?
    parser.add_argument('--num_workers', type=int, default=4) 
    
    #model setting
    parser.add_argument('--gender',type=str, default='')
    parser.add_argument('--pr', type=str, default='bj') #ONLY support bj for now
    parser.add_argument('--cra', action='store_true')
    parser.add_argument('--reginput',type=str, default='dimensional') #if non-cra

    #checkpoints
    parser.add_argument('--ckpt', type=str, default='newckpts/v3_bij.7_afr3_aug3')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)

    #2D detector
    parser.add_argument('--img_crop_scale', type=float, default=0.0)
    parser.add_argument('--bbox_scale', type=float, default=1.2)
    #keys
    parser.add_argument('--vis',  type=str, default='')
    parser.add_argument('--data', type=str, default='h36mp2')#h36mp2, h36mp1, 3dpw, mpi
    parser.add_argument('--wgender', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")
    
    # Model
    regressor, smpl_model = Build_Model(args.batch_size, 
                                        cra_mode=args.cra, 
                                        pr_mode=args.pr, 
                                        device=device, 
                                        itersup=0, 
                                        reginput=args.reginput)
    
    # Data
    pr_path = fetch_processed_pr_path(args.data, args.img_crop_scale, args.bbox_scale)
    img_path= fetch_processed_img_path(args.data, args.img_crop_scale, args.bbox_scale)
    test_w_pr = os.path.exists(pr_path) and os.path.exists(img_path)
    if not test_w_pr:
        pr_path, img_path = '', ''
        args.batch_size = 1
    dataloader = Build_Test_Dataloader(args, pr_path, img_path)

    # Initialize 
    if test_w_pr:
        test_HMR = testHMRPr(regressor, smpl_model, device, args)
    else:
        test_HMR = testHMRImg(regressor, smpl_model,  device, args)
    
    # Test
    if args.start==0 and args.end==0:
        regressor = load_best_checkpoint(regressor, args)
        
        visdir = f'./{args.vis}_{args.dataset}' if args.vis else ''
        mpjpe_pa = test_HMR.test(args.dataset, batch_size=batch_size, visdir=visdir)
    else:
        checkpoints = [f'{args.iterpath}/epoch{n}.tar' for n in range(args.start, args.end)]
        metric_records = []
        for ckpt_path in checkpoints:
            print(ckpt_path)
            if not os.path.exists(ckpt_path):
                continue
            checkpoint = torch.load(ckpt_path, map_location=device)
            new_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if not key.startswith('smpl_model'):
                    new_state_dict[key] = value
            regressor.load_state_dict(new_state_dict, strict=False)
            # regressor.load_state_dict(checkpoint['model_state_dict'])
            metric = test_HMR.test(args.dataset, batch_size=batch_size, visdir='')
            # print()
            metric_records.append(metric)
            print(metric_records)
    
    import ipdb; ipdb.set_trace()