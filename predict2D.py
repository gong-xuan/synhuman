import numpy as np
import os, argparse, sys, cv2
import torch
from torch.utils.data import DataLoader

import configs
from dataload.dataloader import Build_Test_Dataloader

from eval.predict_densepose import setup_densepose_silhouettes, predict_silhouette_densepose
from eval.predict_joints2d import setup_j2d_predictor, predict_joints2D_new


from utils.image_utils import crop_and_resize_iuv_joints2D_torch, crop_bbox_centerscale
from utils.io_utils import write_sample_iuv_j2d, fetch_processed_pr_path, fetch_processed_img_path

sys.path.append(f"{configs.DETECTRON2_PATH}/projects/DensePose")
from densepose.vis.extractor import DensePoseResultExtractor




def debug_pred(dataset, joints2D_predictor, silhouette_predictor, extractor, nlist=[8394]):
    #h36m
    #i1.2_s1.2:[8177, 8429, 8430, 8431, 10016, 10017, 10027, 10039, 10348, 10349, 10355]
    #s1.2:[8394]
    #i2_s1.2:[8429]
    #i1.5_s1.2:[]
    #3dpw
    #i0_s1.2:[17385]
    for n in nlist:
        image = cv2.imread(dataset.imgnames[n])
        center = dataset.bbox_centers[n]
        scale = dataset.scales[n]*1.2
        image = crop_bbox_centerscale(image, center, scale, 
                res=256, resize_interpolation=cv2.INTER_LINEAR)

        #pred IUV 
        with torch.no_grad():
            outputs = silhouette_predictor(image)["instances"]
        # segm = outputs.get("pred_densepose").fine_segm
        bboxes = outputs.get("pred_boxes").tensor
        densepose = extractor(outputs[int(0)])[0][0]
        imap = densepose.labels/24.0
        uvmap = densepose.uv
        uvmap = torch.clip(uvmap, min=0, max=1)
        # import ipdb; ipdb.set_trace()
        iuv = torch.cat([imap[None], uvmap])#(3,h,w)

        #pred j2d
        outputs = joints2D_predictor(image)  # Multiple bboxes + keypoints predictions if there are multiple people in the image
        bboxes = outputs['instances'].pred_boxes.tensor
        keypoints = outputs['instances'].pred_keypoints.cpu().numpy()
        import ipdb; ipdb.set_trace()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='h36mp2') #[h36mp1, h36mp2, ssp3d, 3dpw, mpi]
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--img_crop_scale', type=float, default=0) #1.5
    parser.add_argument('--bbox_scale', type=float, default=1.2)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4) 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # device = torch.device("cuda:0")
    
    #model
    joints2D_predictor = setup_j2d_predictor()
    silhouette_predictor = setup_densepose_silhouettes()
    extractor = DensePoseResultExtractor()
    
    #data 
    dataloader, _, _ = Build_Test_Dataloader(args)
    
    save_pr_dir = fetch_processed_pr_path(args.data, args.img_crop_scale, args.bbox_scale)
    save_img_dir = fetch_processed_img_path(args.data, args.img_crop_scale, args.bbox_scale)

    ########
    # debug_pred(dataset, joints2D_predictor, silhouette_predictor, extractor)
    
    skip_idx = []
    black_idx = []
    count=0
    for n_sample, samples_batch in enumerate(dataloader):
        imagename = samples_batch['imgname'][0]
        image = samples_batch['image'][0].numpy()
        center = samples_batch['center'][0].numpy()
        scale = samples_batch['scale'][0].item()

        IUV = predict_silhouette_densepose(image, silhouette_predictor, extractor, center, scale)#(img, imgw, 3)
        joints2D = predict_joints2D_new(image, joints2D_predictor, center, scale)#(17,3)
        if (IUV is None) or (joints2D is None):
            skip_idx.append(n_sample)
            continue
        IUV, joints2D, cropped_img = crop_and_resize_iuv_joints2D_torch(IUV, 
                                                                configs.REGRESSOR_IMG_WH, 
                                                                joints2D=joints2D, 
                                                                image=image, 
                                                                bbox_scale_factor=args.bbox_scale)
        ##
        bodymask = (24*IUV[:,:,0]).round().cpu().numpy()
        fg_ids = np.argwhere(bodymask != 0) 
        if fg_ids.shape[0]<256:
            print(f'{n_sample}only has {fg_ids.shape[0]} body pixels')
            black_idx.append(n_sample)
            continue
        # bodymask = np.argwhere(IUV[:,:,0].cpu().numpy()!=0)
        # x0,y0 = np.amin(bodymask,axis=0)
        # x1,y1 = np.amax(bodymask,axis=0)
        
        savename = imagename.split('/')[-1][:-4]
        write_sample_iuv_j2d(IUV.numpy(), joints2D, savename, save_pr_dir)
        cv2.imwrite(f'{save_img_dir}/{savename}.png', cropped_img)

        count+=1
        if count%1000==0:
            print(f'Saved {savename}..[{count}/{len(dataloader)}]')
    
    print('--------------------Completed------------------------------------')  
    print('Skip:', skip_idx)  
    print('Black:', black_idx)  
