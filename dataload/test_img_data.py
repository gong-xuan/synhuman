import numpy as np
import cv2
from torch.utils.data import Dataset

import configs
from utils.image_utils import pad_to_square, crop_bbox_centerscale


class TestImg_SSP3D(Dataset):
    def __init__(self, imgpath=configs.SSP3D_IMG_PATH, gtfile=configs.SSP3D_GT, proxy_rep_input_wh=512, crop=0):
        rootpath = '/'.join(imgpath.split('/')[:-1])
        # iuvpath = f'{rootpath}/iuv'
        data =  np.load(gtfile, allow_pickle=True)
        # import ipdb; ipdb.set_trace()
        imgfiles = data['fnames'].tolist()
        self.imgnames = [f'{imgpath}/{imgf}' for imgf in imgfiles]
        # self.iuvnames = [f'{iuvpath}/{imgf[:-4]}.npz' for imgf in imgfiles]
        self.pose = data['poses']
        self.shape = data['shapes']
        self.prep_wh = proxy_rep_input_wh
        self.crop = crop
        self.bbox_centers = data['bbox_centres'] #311*2
        self.scales = data['bbox_whs']

    def __len__(self):
        return len(self.imgnames)

    def crop_resize(self, arr, x0,y0,x1,y1):
        # x0,y0,x1,y1 = bbox_xyxy
        # x0,y0,x1,y1 = round(x0),round(y0),round(x1),round(y1)
        crop_arr = arr[ y0:y1, x0:x1,:]
        crop_arr = pad_to_square(crop_arr)
        resize_arr = cv2.resize(crop_arr, (self.prep_wh, self.prep_wh), interpolation = cv2.INTER_NEAREST)
        return resize_arr

    def process_iuv(self, index): #scale>=1
        imgname = self.imgnames[index]
        fname = self.iuvnames[index]
        data = np.load(fname)
        # bbox_xyxy = data['bbox_xyxy']
        iuv = data['iuv'] #(3, imgh,imgw)
        # print(np.unique(iuv))
        scale = self.scale
        resize_wh = int(self.prep_wh/max(scale,1))
        #
        iuv = pad_to_square(np.transpose(iuv,(1,2,0))) #(max_hw, max_hw, 3)
        # iuv = cv2.resize(iuv, (self.prep_wh, self.prep_wh), interpolation = cv2.INTER_NEAREST)
        iuv = cv2.resize(iuv, (resize_wh, resize_wh), interpolation = cv2.INTER_NEAREST)
        # print(np.unique(iuv))
        image = cv2.imread(imgname)
        # import ipdb; ipdb.set_trace()
        # image = image[y0:y1,x0:x1,:]
        image = pad_to_square(image)
        # image = cv2.resize(image, (self.prep_wh, self.prep_wh), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (resize_wh, resize_wh), interpolation=cv2.INTER_LINEAR)
        
        if scale>1:
            border_wh = (self.prep_wh-resize_wh)//2
            iuv_2 = np.zeros(((self.prep_wh, self.prep_wh, 3)))
            iuv_2[border_wh:border_wh+resize_wh, border_wh:border_wh+resize_wh, :] = iuv
            iuv = iuv_2

            image_2 =  np.zeros(((self.prep_wh, self.prep_wh, 3)))
            image_2[border_wh:border_wh+resize_wh, border_wh:border_wh+resize_wh, :] = image
            image = image_2

        masked_image = image.copy()
        body_mask = (iuv[:,:,0]>0).astype('uint8')
        body_mask = np.expand_dims(body_mask, axis=2)
        masked_image = masked_image*body_mask+(1-body_mask)*np.zeros_like(masked_image)

        # image = torch.from_numpy(image.permute(2,0,1))
        # iuv = torch.from_numpy(iuv.astype(np.float32))  
        # iuv = expand_bbox(iuv,self.prep_wh, self.prep_wh, bbox_xyxy)#(3, wh, wh)
        # iuv = iuv.permute(1,2,0)

        body_pixels = np.argwhere(body_mask[:,:,0] != 0)
        # import ipdb; ipdb.set_trace()
        y0, x0 = np.amin(body_pixels, axis=0)
        y1, x1 = np.amax(body_pixels, axis=0)
        x1, y1 = x1+1, y1+1
        bbox = np.array([x0,y0,x1, y1])
        if self.bbox:
            image = self.crop_resize(image, x0,y0,x1,y1)
            masked_image = self.crop_resize(masked_image, x0,y0,x1,y1)
            iuv = self.crop_resize(iuv, x0,y0,x1,y1)
        return imgname, image, masked_image, iuv, bbox
    
    def __getitem__(self, index):
        imgname = self.imgnames[index]
        image = cv2.imread(imgname)
        # iuv = np.load(self.iuvnames[index],allow_pickle=True)['iuv'] #(3, imgh,imgw)
        # iuv = np.transpose(iuv,(1,2,0)) #(imgh, imgw, 3)
        # assert iuv.shape == image.shape
        # print(np.unique(iuv))
        center = self.bbox_centers[index]
        scale = self.scales[index]
        if self.crop>0:
            scale = scale*self.crop
            image = crop_bbox_centerscale(image, center, scale, 
                res=self.prep_wh, resize_interpolation=cv2.INTER_LINEAR)
            # print('image', index)
            # iuv = crop_bbox_centerscale(iuv, center, scale, 
            #     res=self.prep_wh, resize_interpolation=cv2.INTER_NEAREST)
            scale = 0
        # else: 
        #     image = pad_to_square(image)
        #     iuv = pad_to_square(iuv)

        pose = self.pose[index]
        shape = self.shape[index]
        assert pose.shape == (72,) and shape.shape == (10,), \
            "Poses and shapes are wrong: {}, {}".format(pose.shape, shape.shape)
        return {'imgname': imgname,
                'image':image,
                # 'iuv': iuv,
                'pose': pose,
                'shape': shape,
                'center': center,
                'scale': scale}

        
class TestImg_3DPW(TestImg_SSP3D):
    def __init__(self, rootpath=configs.D3PW_IMG_PATH, gtfile=configs.D3PW_GT, proxy_rep_input_wh=256, crop=0):
        data =  np.load(gtfile, allow_pickle=True)
        imgfiles = data['imgname'].tolist()
        self.imgnames = [f'{rootpath}/{imgf}' for imgf in imgfiles]
        # import ipdb; ipdb.set_trace()
        # self.iuvnames = [rootpath+'/iuvFiles/'+ '/'.join(imgf.split('/')[1:])[:-4] +'.npz' for imgf in imgfiles]
        
        self.pose = data['pose']
        self.shape = data['shape']
        self.prep_wh = proxy_rep_input_wh
        self.crop = crop
        self.bbox_centers = data['center'] #35515*2
        self.scales = data['scale']*200 #35515

class TestImg_H36M(TestImg_SSP3D):
    def __init__(self, protocal=2, rootpath=configs.H36M_IMG_PATH, proxy_rep_input_wh=512, crop=0):
        gtfile=configs.H36M_P2_GT if protocal==2 else configs.H36M_P1_GT
        data = np.load(gtfile, allow_pickle=True)  
        J24_4d = data['S'] #(27558,24,4)
        self.J14_3d = self.root_centered(J24_4d)
        # import ipdb; ipdb.set_trace()
        self.prep_wh = proxy_rep_input_wh
        self.crop = crop
        imgfiles =data['imgname'].tolist()
        self.imgnames = [f'{rootpath}/{imgf}' for imgf in imgfiles]
        # self.iuvnames = [f'{rootpath}/iuv/{imgf[7:-4]}.npz' for imgf in imgfiles]
        self.bbox_centers = data['center'] #27588*2
        self.scales = data['scale']*200 #27588
        # if not usedata=='all':
        #     usen = int(usedata)
        #     self.J14_3d = self.J14_3d[:usen]
        #     self.imgnames = self.imgnames[:usen]
        #     self.bbox_centers = self.bbox_centers[:usen]
        #     self.scales = self.scales[:usen]


    def root_centered(self, S):
        J24_to_J14 = [0,1,2,3,4,5,6,7,8,9,10,11,17,18]
        # joints_name = ("R_Ankle0", "R_Knee1", "R_Hip2", "L_Hip3", "L_Knee4", "L_Ankle5", "R_Wrist6",
        #         "R_Elbow7", "R_Shoulder8", "L_Shoulder9", "L_Elbow10", "L_Wrist11", "Thorax12",
        #         "Head13", "HeadTop14")
        # J24_to_J14 = config.H36M_TO_J14
        
        center = (S[:,2,:3] +  S[:,3,:3])/2 #between two hip points
        center = np.expand_dims(center, axis=1)
        S = S[:, J24_to_J14, :3]
        S = S - center
        return S

    def __getitem__(self, index):
        j14_3d = self.J14_3d[index]
        imgname = self.imgnames[index]
        image = cv2.imread(imgname)
        # iuv = np.load(self.iuvnames[index],allow_pickle=True)['iuv'] #(3, imgh,imgw)
        # iuv = np.transpose(iuv,(1,2,0)) #(imgh, imgw, 3)
        # assert iuv.shape == image.shape
        # print(np.unique(iuv))
        center = self.bbox_centers[index]
        scale = self.scales[index]

        if self.crop!=0:
            scale = scale*self.crop
            image = crop_bbox_centerscale(image, center, scale, 
                res=self.prep_wh, resize_interpolation=cv2.INTER_LINEAR)
            # print('image', index)
            # iuv = crop_bbox_centerscale(iuv, center, scale, 
            #     res=self.prep_wh, resize_interpolation=cv2.INTER_NEAREST)
            # print('iuv', index)
            scale = 0
        # else: #make it a square
        #     image = pad_to_square(image)
        #     iuv = pad_to_square(iuv)
        
        # masked_image = image.copy()
        # body_mask = (iuv[:,:,0]>0).astype('uint8')
        # body_mask = np.expand_dims(body_mask, axis=2)
        # masked_image = masked_image*body_mask+(1-body_mask)*np.zeros_like(masked_image)

        # image = torch.from_numpy(image.permute(2,0,1))
        # iuv = torch.from_numpy(iuv.astype(np.float32)) 
        # pose = torch.from_numpy(pose.astype(np.float32))
        # shape = torch.from_numpy(shape.astype(np.float32))
        return {'imgname': imgname,
                'image':image,
                # 'iuv': iuv,
                'j14_3d': j14_3d,
                'center': center,
                'scale': scale}

class TestImg_MPI3DHP(TestImg_H36M):
    def __init__(self, gtfile=configs.MPI3DHP_GT, rootpath=configs.MPI3DHP_IMG_PATH, proxy_rep_input_wh=512, crop=0):
        data = np.load(gtfile, allow_pickle=True)  
        J24_4d = data['S'] #(27558,24,4)
        self.J14_3d = self.root_centered(J24_4d)
        # import ipdb; ipdb.set_trace()
        self.prep_wh = proxy_rep_input_wh
        self.crop = crop
        imgfiles =data['imgname'].tolist()
        self.imgnames = [f'{rootpath}/{imgf}' for imgf in imgfiles]
        # self.iuvnames = [f'{rootpath}/iuv/{imgf[7:-4]}.npz' for imgf in imgfiles]
        self.bbox_centers = data['center'] 
        self.scales = data['scale']*200 
 