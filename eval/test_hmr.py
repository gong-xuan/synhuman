import tqdm
import torch
import numpy as np

from eval.metrics import AverageMeter
from eval.predict_densepose import setup_densepose_silhouettes, predict_silhouette_densepose
from eval.predict_joints2d import setup_j2d_predictor, predict_joints2D_new
from model.model import Build_SMPL
import eval.metrics as metrics
import utils.label_conversions as LABELCONFIG
from utils.smpl_utils import smpl_forward

import sys, configs
sys.path.append(f"{configs.DETECTRON2_PATH}/projects/DensePose")
from densepose.vis.extractor import DensePoseResultExtractor

class testHMRImg():
    def __init__(self, 
                regressor, 
                smpl_model, 
                device, 
                args, 
                eval_j14=False):

        self.regressor = regressor
        self.smpl_model = smpl_model
        self.device = device
        self.pr_mode = args.pr
        self.pr_wh = configs.REGRESSOR_IMG_WH
        self.eval_j14 = eval_j14

        if args.wgender: #only SSP3D has valid gender #NOT valid for now
            self.smpl_model_male = Build_SMPL(self.batch_size, self.device, gender='male')
            self.smpl_model_female = Build_SMPL(self.batch_size, self.device, gender='female')
        self.wgender = args.wgender
        
        self.set_detector2D()


    def set_detector2D(self):
        # Set-up proxy representation predictors.
        self.silhouette_predictor = setup_densepose_silhouettes()
        self.extractor = DensePoseResultExtractor()
        self.joints2D_predictor = setup_j2d_predictor()

    
    def get_proxy_rep(self, samples_batch):#batch_size=1
        image = samples_batch['image'][0].numpy()
        center = samples_batch['center'][0].numpy()
        scale = samples_batch['scale'][0].item()
        IUV = predict_silhouette_densepose(image, self.silhouette_predictor, self.extractor, center, scale)#(3,h,w) torch
        joints2D = predict_joints2D_new(image, self.joints2D_predictor, center, scale)
        # import ipdb; ipdb.set_trace()
        self.IUV = torch.tensor(IUV).permute(1,2,0)[None].to(self.device)#(bs,h,w,3)
        self.joints2D = torch.tensor(joints2D[:,:2])[None].to(self.device)#(bs,17,2)
        if (IUV is None) or (joints2D is None):
            return None, None, None
        else:
            return self.cal_proxy_rep(image, IUV, joints2D)

    def cal_proxy_rep(self, image, IUV, joints2D):
        IUV = IUV.to(self.device)
        if self.pr_mode=='bj':    
            
            IUV, joints2D, image = crop_and_resize_iuv_joints2D_torch(IUV, self.pr_wh, joints2D=joints2D, 
            image=image, bbox_scale_factor=self.bbox_scale_factor)
            target_partseg = (IUV[:,:,0]*24).round() 
            silhouette = convert_multiclass_to_binary_labels_torch(target_partseg).cpu().numpy()#(H,W)
            # Create proxy representation
            heatmaps = convert_2Djoints_to_gaussian_heatmaps(joints2D.astype(np.int16),
                                                     self.pr_wh)
            # import ipdb; ipdb.set_trace()
            proxy_rep = np.concatenate([silhouette[:, :, None], heatmaps], axis=-1)
            proxy_rep = np.transpose(proxy_rep, [2, 0, 1])  # (C, out_wh, out_WH)
            proxy_rep = proxy_rep[None]  # add batch dimension
            proxy_rep = torch.from_numpy(proxy_rep).float().to(self.device)
            #vis
            binaryseg = silhouette[:,:,None]
            binaryseg = np.concatenate([binaryseg,binaryseg,binaryseg],axis=2) 
            body_mask = binaryseg
            vis_proxy_rep = body_mask*binaryseg.astype('uint8')*255+(1-body_mask)*image
            vis_proxy_rep = saveKP2D(joints2D, None, image=vis_proxy_rep, H=self.pr_wh, W=self.pr_wh, color=(0,255,0), addText=False)
        

        
        return image, proxy_rep, vis_proxy_rep

    def forward_batch(self, proxy_rep):
        with torch.no_grad():
            if hasattr(self.regressor, 'add_channels'):
                if torch.tensor(self.regressor.add_channels).bool().any().item():
                    # import ipdb; ipdb.set_trace()
                    self.regressor.gt_IUV = self.IUV
                    # self.regressor.gt_IUV_mask = None
                    self.regressor.gt_joints2d_coco = self.joints2D
            
            pred_cam_wp, pred_pose, pred_shape, _ = self.regressor(proxy_rep)
            if VIS:
                self.pred_cam_wp_list = pred_cam_wp
                self.pred_pose_list = pred_pose
                self.pred_shape_list = pred_shape

            pred_cam_wp, pred_pose, pred_shape = pred_cam_wp[0], pred_pose[0], pred_shape[0]
            # Convert pred pose to rotation matrices
            if pred_pose.shape[-1] == 24 * 3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
            elif pred_pose.shape[-1] == 24 * 6:
                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            # import ipdb; ipdb.set_trace()
            # pred_shape[0][1]=0.02
            pred_vertices, pred_joints_all = self.renderSyn.smpl_model(body_pose=pred_pose_rotmats[:, 1:], #[1,23,3,3]
                                global_orient=pred_pose_rotmats[:, 0].unsqueeze(1), #[1,1,3,3]
                                betas=pred_shape,#[1,10]
                                pose2rot=False)
        
            pred_joints_h36m = pred_joints_all[:, config.ALL_JOINTS_TO_H36M_MAP, :]
            if self.use_j14:
                pred_joints_h36mlsp = pred_joints_h36m[:, config.H36M_TO_J14, :]
            else:
                pred_joints_h36mlsp = pred_joints_h36m[:, config.H36M_TO_J17, :]
            pred_reposed_vertices, _ = self.renderSyn.smpl_model(betas=pred_shape)
        return  pred_joints_h36mlsp, pred_reposed_vertices, pred_vertices, pred_cam_wp


    def update_metrics_step(self, samples_batch, proxy_rep_batch, printt=False):
        #forward
        pred_joints_h36mlsp, pred_reposed_vertices, pred_vertices, pred_cam_wp = self.forward_batch(proxy_rep_batch)
        
        #GT
        if self.withshape:
            target_pose = samples_batch['pose'].to(self.device).float()#bx72
            target_shape = samples_batch['shape'].to(self.device).float()#bx10
            target_pose_rotmats, target_vertices, joints_all, target_reposed_vertices, _ = smpl_forward(
                target_shape, 
                target_pose,
                self.smpl_model)

            target_joints_h36m = joints_all[:, LABELCONFIG.ALL_JOINTS_TO_H36M_MAP, :]
            
            if self.eval_j14:
                target_joints_h36mlsp = target_joints_h36m[:, LABELCONFIG.H36M_TO_J14, :]
            else:
                target_joints_h36mlsp = target_joints_h36m[:, LABELCONFIG.H36M_TO_J17, :]
        else:
            target_joints_h36mlsp = samples_batch['j17_3d'].to(self.device).float()
            if self.eval_j14:
                target_joints_h36mlsp = target_joints_h36mlsp[:,:14]
            target_vertices = None
        self.target_vertices = target_vertices
        # origin at center
        pred_joints_h36mlsp = pred_joints_h36mlsp - (pred_joints_h36mlsp[:,[2],:]+pred_joints_h36mlsp[:,[3],:])/2
        target_joints_h36mlsp = target_joints_h36mlsp- (target_joints_h36mlsp[:,[2],:]+target_joints_h36mlsp[:,[3],:])/2
        # no use?
        # pred_vertices = pred_vertices - (pred_joints_h36mlsp[:,[2],:]+pred_joints_h36mlsp[:,[3],:])/2
        # target_vertices = target_vertices - (target_joints_h36mlsp[:,[2],:]+target_joints_h36mlsp[:,[3],:])/2
        #
        #metric
        batch_size = pred_joints_h36mlsp.shape[0]
        if 'mpjpe_pa' in self.metrics_track:
            mpjpe_pa = metrics.cal_mpjpe_pa(pred_joints_h36mlsp.detach().cpu().numpy(), 
                                            target_joints_h36mlsp.detach().cpu().numpy())
            self.mpjpe_pa.update(mpjpe_pa, n=batch_size)
            if printt:
                print(f'mpjpe_pa for {self.mpjpe_pa.count}: {self.mpjpe_pa.average()}')
        
        if 'mpjpe_sc' in self.metrics_track:
            mpjpe_sc = metrics.cal_mpjpe_sc(pred_joints_h36mlsp.detach().cpu().numpy(), 
                                            target_joints_h36mlsp.detach().cpu().numpy())
            self.mpjpe_sc.update(mpjpe_sc, n=batch_size)
            if printt:
                print(f'mpjpe_sc for {self.mpjpe_sc.count}: {self.mpjpe_sc.average()}')
        
        if 'mpjpe' in self.metrics_track:
            mpjpe = metrics.cal_mpjpe(pred_joints_h36mlsp.detach().cpu().numpy(), 
                                    target_joints_h36mlsp.detach().cpu().numpy())
            self.mpjpe.update(mpjpe, n=batch_size)
            if printt:
                print(f'mpjpe for {self.mpjpe.count}: {self.mpjpe.average()}')

        if 'pve-ts_sc' in self.metrics_track:
            pve = metrics.cal_pve_ts_sc(pred_reposed_vertices.detach().cpu().numpy(), 
                                        target_reposed_vertices.detach().cpu().numpy())
            self.pve_t_sc.update(pve,  n=batch_size)
            if printt:
                print(f'pve_t_sc for {self.pve_t_sc.count}: {self.pve_t_sc.average()}') 

        if 'pve' in self.metrics_track:
            pve = metrics.cal_pve(pred_vertices.detach().cpu().numpy(), 
                                target_vertices.detach().cpu().numpy())
            self.pve.update(pve,  n=batch_size)
            if printt:
                print(f'pve for {self.pve.count}: {self.pve.average()}') 

        if 'pve_pa' in self.metrics_track:
            pve_pa = metrics.cal_pve_pa(pred_vertices.detach().cpu().numpy(), 
                                        target_vertices.detach().cpu().numpy())
            self.pve_pa.update(pve_pa,  n=batch_size)
            if printt:
                print(f'pve_pa for {self.pve_pa.count}: {self.pve_pa.average()}') 

        if 'pck_pa' in self.metrics_track:
            pck_pa, auc_pa = metrics.cal_pck_pa(pred_joints_h36mlsp.detach().cpu().numpy(), 
                                                target_joints_h36mlsp.detach().cpu().numpy())
            self.pck.update(pck_pa, n=batch_size)
            self.auc.update(auc_pa, n=batch_size)
            if printt:
                print(f'pck_pa for {self.pck.count}: {self.pck.average()}')
                print(f'auc_pa for {self.auc.count}: {self.auc.average()}')

        # saveKP2D(joints2D, f'{visdir}/{n_sample}_{self.pr_mode}_j2d.png', image=image)
        # print(mpjpe)
        return None, pred_vertices, pred_cam_wp, target_vertices




    def test(self, dataloader, print_freq=100):
        self.mpjpe_pa = AverageMeter()
        self.mpjpe_sc = AverageMeter()
        self.mpjpe = AverageMeter()
        self.pck = AverageMeter()
        self.auc = AverageMeter()
        self.pve = AverageMeter()
        self.pve_pa = AverageMeter()
        self.pve_t_sc = AverageMeter()
        

        self.regressor.eval()
        for n_sample, samples_batch in enumerate(tqdm(dataloader)):
            images = samples_batch['image'].numpy()
            cropped_image, proxy_rep_batch, vis_proxy_rep = self.get_proxy_rep(samples_batch)
            
            
            if self.usegender:
                self.update_metrics_step_withgender(
                samples_batch, proxy_rep_batch, printt=(n_sample%print_freq==0))
            else:
                self.update_metrics_step(
                    samples_batch, proxy_rep_batch, printt=(n_sample%print_freq==0))
            
            
        # Complete
        if self.mpjpe_pa.count:  
            print(f'mpjpe_pa for {self.mpjpe_pa.count}: {self.mpjpe_pa.average()}')
        if self.mpjpe_sc.count:
            print(f'mpjpe_sc for {self.mpjpe_sc.count}: {self.mpjpe_sc.average()}')
        if self.mpjpe.count:
            print(f'mpjpe for {self.mpjpe.count}: {self.mpjpe.average()}')
        if self.pck.count:
            print(f'pck for {self.pck.count}: {self.pck.average()}')
        if self.auc.count:
            print(f'auc for {self.auc.count}: {self.auc.average()}')
        if self.pve_t_sc.count:
            print(f'pve_t_sc for {self.pve_t_sc.count}: {self.pve_t_sc.average()}')  
        if self.pve.count:
            print(f'pve for {self.pve.count}: {self.pve.average()}')  
        if self.pve_pa.count:
            print(f'pve_pa for {self.pve_pa.count}: {self.pve_pa.average()}')  
                     
        return self.mpjpe_pa.average()

class testHMRPr(testHMRImg):
    
    
    def proxy_rep_vis(self, images_numpy, IUV, joints2D):
        batch_size = images_numpy.shape[0]
        body_masks = ((IUV[:,:,:,0]*24).round()>0).cpu().numpy().astype('uint8')
        vis_proxy_rep_list = []
        for b in range(batch_size):
            body_mask = np.expand_dims(body_masks[b], axis=2)
            image = images_numpy[b]
            if self.pr_mode=='bj':
                proxy_image = 255
            else:#iuv
                bgr = hsv_to_bgr(IUV[b], keep_background=True)
                proxy_image = 255*bgr.cpu().numpy()
            vis_proxy_rep = body_mask*proxy_image+(1-body_mask)*image
            vis_proxy_rep = saveKP2D(joints2D[b].cpu().numpy(), None, image=vis_proxy_rep, H=self.pr_wh, W=self.pr_wh, color=(0,255,0), addText=False)
            vis_proxy_rep_list.append(vis_proxy_rep)
        return vis_proxy_rep_list

    def get_proxy_rep(self, samples_batch):#return batch
        images = samples_batch['image'].numpy()
        IUV = samples_batch['iuv'].to(self.device)
        joints2D = samples_batch['j2d'].int().to(self.device)
        if 'e' in self.pr_mode:
            rgb_in = torch.tensor(images).to(self.device).permute(0,3,1,2).float()
            edge = self.renderSyn.edge_detect_model(rgb_in)['thresholded_thin_edges'] 
        else:
            edge = None
        # import ipdb; ipdb.set_trace()
        self.IUV = IUV##(bs,h,w,3)
        self.joints2D = joints2D#(bs,17,2)
        if AUG_IUV:
            aug_iuv_map = IUV.clone()
            # seg_extreme_crop = random_extreme_crop(orig_segs=(IUV[:,:,:,0]*24).round(),
            #                                     extreme_crop_probability=CROP_PROB)
            seg_extreme_crop =crop_one_coarse_part(orig_segs=(IUV[:,:,:,0]*24).round())
            aug_iuv_map[seg_extreme_crop==0] = torch.tensor([0,0,0]).float().to(self.device)
            IUV = aug_iuv_map
            self.IUV = IUV
        joints2D_vis = torch.ones(joints2D.shape[:2], device=self.device, dtype=torch.bool)
        if AUG_J2D:
            joints2D_vis = random_remove_joints2D(joints2D_vis, REMOVE_JOINTS_INDEX, probability_to_remove=REMOVE_J2D_PROB)
            # joints2D = random_joints2D_deviation(joints2D.float(),
            #                             delta_j2d_dev_range=[-int(DEVIATION),int(DEVIATION)],
            #                             delta_j2d_hip_dev_range=[-int(DEVIATION),int(DEVIATION)])
            joints2D = joints2D.int()
        
        # import ipdb; ipdb.set_trace()
        proxy_rep = convert_to_proxyfeat_batch(self.pr_mode, IUV, joints2D, edge, joints2d_no_occluded_coco=joints2D_vis, mode='test')
        
        return images, proxy_rep, vis_proxy_rep_list