

def get_camK_fast(self, pred_cam):
    focal_length = pred_cam[:,0] #bs
    t_x, t_y= -pred_cam[:,1]*focal_length, -pred_cam[:,2]*focal_length#bs
    focal_length = focal_length[:,None, None]
    t_x = t_x[:,None, None]
    t_y = t_y[:,None, None]
    fill_zeros = torch.zeros_like(t_x)
    fill_ones = torch.ones_like(t_x)
    #cam_K: (bs,3,3)
    cam_K_row1 = torch.cat([focal_length,fill_zeros, t_x], dim=2)#(bs,1,3)
    cam_K_row2 = torch.cat([fill_zeros, focal_length, t_y], dim=2)#(bs,1,3)
    cam_K_row3 = torch.cat([fill_zeros, fill_zeros, fill_ones], dim=2)#(bs,1,3)

    cam_K = torch.cat([cam_K_row1, cam_K_row2, cam_K_row3], dim=1)
    return cam_K

def pred_verts_to_IUV(self,  vertices, pred_cam, REGRESSOR_IMG_WH, cam_R=None, cam_T=None):
    batch_size = 1

    IUV_processed = uv_utils.IUV_Densepose(device=self.device)
    I = IUV_processed.get_I() #(7829,1) 
    U, V = IUV_processed.get_UV() #(7829,1)
    IUVnorm= torch.cat([I/24, U, V], dim=1)
    self.IUVnorm_list = [IUVnorm for _ in range(batch_size)] #(bs, 7829, 3)
    self.blendparam = BlendParams()  

    # cam_K = self.get_camK(pred_cam)
    cam_K = self.get_camK_fast(pred_cam)#verified
    # import ipdb; ipdb.set_trace()
    renderer= P3dRenderer(batch_size, cam_K, cam_R=cam_R, device=self.device, img_wh=REGRESSOR_IMG_WH, 
            render_options={'pseg':0, 'depth':0, 'normal':0, 'iuv':1})    
    ######## forward
    unique_verts = [vertices[:,vid] for vid in renderer.to_1vertex_id] #(7829,bs,3)
    unique_verts = torch.stack(unique_verts, dim=1) #(bs, 7829, 3)
    #convert to camera coordinate R
    unique_verts = persepctive_project(unique_verts, renderer.cam_R_trans, cam_T=None, cam_K=None)

    #mesh
    verts_list = [unique_verts[nb] for nb in range(batch_size)] #(bs, 7829, 3)
    # desired_colors = [torch.ones((7829,3)).float().to(device) for _ in range(batch_size)] #(bs, 7829, 3)
    mesh_batch = Meshes(verts=verts_list, faces=renderer.faces_list, textures =  Textures(verts_rgb=self.IUVnorm_list))##norm I to [0,1]?

    cam_T = renderer.rectify_cam_T(None)
    cam_T[0,2] = 5
    cameras = OrthographicCameras(#PerspectiveCameras(
        focal_length=((renderer.focal_x, renderer.focal_y),),
        principal_point=((renderer.t_x, renderer.t_y),),
        # K=torch.from_numpy(K)[None].float(), 
        R = renderer.cam_R_render,
        # R = None,
        T = cam_T,
        device=self.device, 
        image_size=((-1, -1),)
        # image_size=((img_wh, img_wh),)
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=renderer.raster_settings)
    fragments = rasterizer(mesh_batch)
    # import ipdb; ipdb.set_trace()
    colors = mesh_batch.sample_textures(fragments)#(BS, H, W, 1, 4)
    images = hard_rgb_blend(colors, fragments, self.blendparam)# (BS, H, W, 4)
    I_map = images[:,:,:,0]#24 unique normalized
    # U_map = images[:,:,:,1]#nearest, require interpolation?
    # V_map = images[:,:,:,2]#nearest, require interpolation?
    # import ipdb; ipdb.set_trace()
    return images[:,:,:,:3]#(b,h,w,3)

def vis_mesh_onimg(self, images, nsample):
        joints2D = self.joints2D
        heatmaps = self.heatmaps
        IUV = self.IUV
        batchsize = IUV.shape[0]
        cam_params =  self.pred_cam_wp_list
        pose_params =  self.pred_pose_list
        shape_params =  self.pred_shape_list
        vertices = self.vertices
        # import ipdb; ipdb.set_trace()
        pred_vertices = self.vismesh_from_pred_batch(images, cam_params[0], 
                    pose_params[0], shape_params[0], bgwithimage=True)
        for bs in range(batchsize):
            # pred_IUV = self.pred_verts_to_IUV(vertices[bs][None],cam_params[0][bs][None], config.REGRESSOR_IMG_WH)#(b,h,w,3)
            # iuv_ts = pred_IUV[0].permute(2,0,1)
            # body_mask = (iuv_ts>0).all(dim=0)[:,:,None].int().cpu().numpy()
            # bgr = hsv_to_bgr(iuv_ts.permute(1,2,0), keep_background=True)
            # img_to_blend = 255*bgr.cpu().numpy()
            #######iuv
            # iuv2 =vis_iuv_onimg(image, iuv_ts, savepath=None)
            # iuv1 = vis_iuv_onimg(image, IUV[0].permute(2,0,1), savepath=None)
            # j2d_image = saveKP2D(joints2D[bs].cpu().numpy(), None, image=image,  H=256, W=256, color=(0,255,0), addText=False)
            # allimage =np.concatenate([image,iuv1, j2d_image, iuv2],axis=1)
            # # if self.pred_j3d is not None:
                
            # cv2.imwrite(f'{VISDIR}/{nsample}.png',allimage)
            # mesh_on_image = self.vismesh_from_pred_batch(image, cam_params[0][bs][None], 
            #         pose_params[0][bs][None], shape_params[0][bs][None], bgwithimage=False)  
            # mesh_on_image = mesh_on_image[10:-10,50:-60,:]          
            # cv2.imwrite(f'./cvpr_images/figure1/meshslim0.2.png', mesh_on_image)
            # mesh_on_image = self.vismesh_from_pred_batch(image, cam_params[0][bs][None], 
            #         pose_params[0][bs][None], shape_params[0][bs][None], bgwithimage=True)   
            # mesh_on_image = mesh_on_image[10:-10,50:-60,:]          
                     
            # cv2.imwrite(f'./cvpr_images/figure1/meshslim0.2_w.png', mesh_on_image)
            # import ipdb; ipdb.set_trace()

            # import ipdb; ipdb.set_trace()
            # self.count +=1
            self.count = nsample
            # if self.count%VIS_INTERVAL!=0:
            #     continue
            ###############################################
            image = images[bs]
            image = black_to_write(image)
            cv2.imwrite(f'{VISDIR}/img{self.count}.png', image) 
            # if AUG_IUV:
            #     iuv1 = vis_iuv_onimg(image, IUV[bs].permute(2,0,1), savepath=f'{VISDIR}/img{self.count}_m{MODELID}IUV.png')
            # else:
            #     iuv1 = vis_iuv_onimg(image, IUV[bs].permute(2,0,1), savepath=f'{VISDIR}/img{self.count}_IUV.png')
            mesh_image, mesh_img_fuse = visualize_mesh_on_img(image, pred_vertices[bs][None].detach(), self.pr_wh, self.device, 
                cam_params[0][bs][None].cpu().detach().numpy(), bbox=None, fusewimg=True, color=colorsets[1], rtboth=True)
            cv2.imwrite(f'{VISDIR}/img{self.count}_p.png', mesh_image) 
            cv2.imwrite(f'{VISDIR}/img{self.count}_pfuse.png', mesh_img_fuse) 
            
            mesh_image, mesh_img_fuse = visualize_mesh_on_img(image, self.target_vertices[bs][None].detach(), self.pr_wh, self.device, 
                cam_params[0][bs][None].cpu().detach().numpy(), bbox=None, fusewimg=True, color=colorsets[MODELID], rtboth=True)
            cv2.imwrite(f'{VISDIR}/img{self.count}_t.png', mesh_image) 
            cv2.imwrite(f'{VISDIR}/img{self.count}_tfuse.png', mesh_img_fuse) 
            # j2d_image = saveKP2D(joints2D[bs].cpu().numpy(), None, image=image,  H=256, W=256, color=(0,255,0), addText=False)
            # cv2.imwrite(f'{VISDIR}/img{self.count}_j2d.png', j2d_image) 
            # import ipdb; ipdb.set_trace()
            # if self.count<10:
            #     savestr = f'0000{self.count}'
            # elif self.count<100:
            #     savestr = f'000{self.count}'
            # elif self.count<1000:
            #     savestr = f'00{self.count}'
            # elif self.count<10000:
            #     savestr = f'0{self.count}'
            # else:
            #     savestr = f'{self.count}'
            # cv2.imwrite(f'{VISDIR}/{savestr}.png', mesh_on_image) 
            # print(f'Saved..{savestr}')


def vis_pr_only(self, image):
    joints2D = self.joints2D
    IUV = self.IUV
    batchsize = IUV.shape[0]
    cam_params =  self.pred_cam_wp_list
    pose_params =  self.pred_pose_list
    shape_params =  self.pred_shape_list
    pathdir = './cvpr_images/ssp3dpr'
    for bs in range(batchsize):
        
        self.count +=1
        cv2.imwrite(f'{pathdir}/{self.count}_img.png', image)
        bgr = hsv_to_bgr(IUV[bs], keep_background=True)
        image2 = 255*bgr.cpu().numpy()
        cv2.imwrite(f'{pathdir}/{self.count}_iuv.png', image2)
        image3=saveKP2D(joints2D[bs].cpu().numpy(), f'{pathdir}/{self.count}_j2d.png', image=None, addText=False)
        body_mask = ((IUV[bs,:,:,0]*24).round()>0).cpu().numpy().astype('uint8')#0,1
        body_mask = np.expand_dims(body_mask*255, axis=2)
        body_image = np.concatenate([body_mask,body_mask,body_mask],axis=2)
        # import ipdb; ipdb.set_trace()  
        image3=saveKP2D(joints2D[bs], f'{pathdir}/{self.count}_bj2d.png', image=body_image, addText=False)
        print(self.count)
    # import ipdb; ipdb.set_trace()  

def vis_pr_mesh(self, image, vis_proxy_rep, pred_vertices, pred_cam_wp, gt_vertices, savepath, bbox=None, 
            pred_joints_h36mlsp = None, target_joints_h36mlsp = None):
        normal_image = visualize_mesh_on_img(image, pred_vertices, self.pr_wh, self.device, pred_cam_wp.cpu().detach().numpy(), bbox=bbox)
        # inter_image = inter_rep_vis(image, iuv, joints2D, self.pr_wh, self.pr_mode, body_mask=None)
        allimage = np.concatenate([image, vis_proxy_rep, normal_image], axis=1)
        if gt_vertices is not None:
            gt_normal_image = visualize_mesh_on_img(image, gt_vertices.detach(), self.pr_wh, device, pred_cam_wp.cpu().detach().numpy(), bbox=bbox)
            allimage = np.concatenate([allimage, gt_normal_image], axis=1)
        cv2.imwrite(savepath, allimage)
        if pred_joints_h36mlsp is not None and target_joints_h36mlsp is not None:
            visualize_j3d(pred_joints_h36mlsp[0].cpu().numpy().tolist(), target_joints_h36mlsp[0].detach().cpu().numpy().tolist(), 
                f'{savepath}_j3d.png')

    

    def vismesh_from_pred_batch(self, image, pred_cam_wp, pred_pose, pred_shape, fusewimg=False, bgwithimage=False):
        if pred_pose.shape[-1] == 24 * 3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
        elif pred_pose.shape[-1] == 24 * 6:
            pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

        pred_vertices, pred_joints_all = self.renderSyn.smpl_model(body_pose=pred_pose_rotmats[:, 1:], #[1,23,3,3]
                            global_orient=pred_pose_rotmats[:, 0].unsqueeze(1), #[1,1,3,3]
                            betas=pred_shape,#[1,10]
                            pose2rot=False)
        # if bgwithimage:
        #     white_image = image
        # else:
        #     white_image = (np.ones_like(image)*255).astype('uint8')
        
        return pred_vertices

    def vis_heatmap_iuv_mesh(self):
        heatmaps = self.heatmaps
        IUV = self.IUV
        batchsize = IUV.shape[0]
        cam_params =  self.pred_cam_wp_list
        pose_params =  self.pred_pose_list
        shape_params =  self.pred_shape_list
        # cam_params = self.regressor.cam_params
        # pose_params = self.regressor.pose_params
        # shape_params = self.regressor.shape_params
        # aligned_j2d = self.regressor.aligned_j2d
        # aligned_i = self.regressor.aligned_i
        # aligned_uv= self.regressor.aligned_uv

        for bs in range(batchsize):
            self.count +=1
            pathdir = f'{VISDIR}/group{self.count}'
            if not os.path.isdir(pathdir):
                os.makedirs(pathdir)
            heatmap = heatmaps[bs].cpu().numpy()
            iuv_ts = IUV[bs]
            heatimg = np.amax(heatmap, axis=0)#(17,h,w)
            image1 = cv2.applyColorMap((heatimg*255).astype('uint8'), cv2.COLORMAP_JET)
            cv2.imwrite(f'{pathdir}/joints.png', image1)
            #
            for nj in range(17):
                heatimg = heatmap[nj]
                image_j = cv2.applyColorMap((heatimg*255).astype('uint8'), cv2.COLORMAP_JET)
                cv2.imwrite(f'{pathdir}/j{nj+1}.png', image_j)
            #
            bgr = hsv_to_bgr(iuv_ts, keep_background=True)
            image2 = 255*bgr.cpu().numpy()
            cv2.imwrite(f'{pathdir}/iuv.png', image2)
            iuv_numpy = iuv_ts.cpu().numpy()
            i_image = cv2.applyColorMap((iuv_numpy[:,:,0]*255).astype('uint8'), cv2.COLORMAP_JET)
            u_image = cv2.applyColorMap((iuv_numpy[:,:,1]*255).astype('uint8'), cv2.COLORMAP_JET)
            v_image = cv2.applyColorMap((iuv_numpy[:,:,2]*255).astype('uint8'), cv2.COLORMAP_JET)
            cv2.imwrite(f'{pathdir}/i.png', i_image)
            cv2.imwrite(f'{pathdir}/u.png', u_image)
            cv2.imwrite(f'{pathdir}/v.png', v_image)
            mesh_image = self.vismesh_from_pred_batch(image1, cam_params[0][bs][None], 
                    pose_params[0][bs][None], shape_params[0][bs][None], bgwithimage=False)
            cv2.imwrite(f'{pathdir}/mesh.png', mesh_image)
            ########with image
            #mesh
            # import ipdb; ipdb.set_trace()
            # for nmesh in range(3):
            #     mesh_image = self.vismesh_from_pred_batch(image1, cam_params[nmesh][bs][None], 
            #         pose_params[nmesh][bs][None], shape_params[nmesh][bs][None], fusewimg=False)
            #     cv2.imwrite(f'{pathdir}/mesh{nmesh}.png', mesh_image)
            # for nalign in range(2):
            #     heatmap = aligned_j2d[nalign][bs].detach().cpu().numpy()
            #     heatimg = np.amax(heatmap, axis=0)#(17,h,w)
            #     image1 = cv2.applyColorMap((heatimg*255).astype('uint8'), cv2.COLORMAP_JET)
            #     cv2.imwrite(f'{pathdir}/meshjoints{nalign+2}.png', image1)

        # import ipdb; ipdb.set_trace()
                
    def vis_pr_mesh_onimg(self, image):
        joints2D = self.joints2D
        heatmaps = self.heatmaps
        IUV = self.IUV
        batchsize = IUV.shape[0]
        cam_params =  self.pred_cam_wp_list
        pose_params =  self.pred_pose_list
        shape_params =  self.pred_shape_list
        # import ipdb; ipdb.set_trace()

        for bs in range(batchsize):
            self.count +=1
            pathdir = f'{VISDIR}/group{self.count}'
            if not os.path.isdir(pathdir):
                os.makedirs(pathdir)
            heatmap = heatmaps[bs].cpu().numpy()
            iuv_ts = IUV[bs]
            heatimg = np.amax(heatmap, axis=0)#(17,h,w)
            image1 = cv2.applyColorMap((heatimg*255).astype('uint8'), cv2.COLORMAP_JET)
            cv2.imwrite(f'{pathdir}/joints.png', image1)
            bgr = hsv_to_bgr(iuv_ts, keep_background=True)
            image2 = 255*bgr.cpu().numpy()
            cv2.imwrite(f'{pathdir}/iuv.png', image2)
            iuv_numpy = iuv_ts.cpu().numpy()
            i_image = cv2.applyColorMap((iuv_numpy[:,:,0]*255).astype('uint8'), cv2.COLORMAP_JET)
            u_image = cv2.applyColorMap((iuv_numpy[:,:,1]*255).astype('uint8'), cv2.COLORMAP_JET)
            v_image = cv2.applyColorMap((iuv_numpy[:,:,2]*255).astype('uint8'), cv2.COLORMAP_JET)
            cv2.imwrite(f'{pathdir}/i.png', i_image)
            cv2.imwrite(f'{pathdir}/u.png', u_image)
            cv2.imwrite(f'{pathdir}/v.png', v_image)
            mesh_image = self.vismesh_from_pred_batch(image1, cam_params[0][bs][None], 
                    pose_params[0][bs][None], shape_params[0][bs][None], bgwithimage=False)
            cv2.imwrite(f'{pathdir}/mesh.png', mesh_image)   
            ###############################################
            cv2.imwrite(f'{pathdir}/img.png', image) 
            mesh_on_image = self.vismesh_from_pred_batch(image, cam_params[0][bs][None], 
                    pose_paprams[0][bs][None], shape_params[0][bs][None], bgwithimage=True)
            iuv_image = vis_iuv_onimg(image, iuv_ts.permute(2,0,1))
            j2d_image = saveKP2D(joints2D[bs].cpu().numpy(), None, image=image,  H=256, W=256, color=(0,255,0), addText=False)
            
            cv2.imwrite(f'{pathdir}/img_mesh.png', mesh_on_image) 
            cv2.imwrite(f'{pathdir}/img_j2d.png', j2d_image) 
            cv2.imwrite(f'{pathdir}/img_iuv.png', iuv_image) 
            # import ipdb; ipdb.set_trace()