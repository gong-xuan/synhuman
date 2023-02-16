import os, cv2
import numpy as np
from torch.utils.data import Dataset
import configs

class TestPr_H36M(Dataset):
    def __init__(self,protocal=2):
        if protocal==2:
            imgpath=configs.H36M_IMG_DIR
            prpath=configs.H36M_PR_DIR
            gtfile=configs.H36M_GT
        elif protocal==1:
            imgpath=configs.H36M_P1_IMG_DIR
            prpath=configs.H36M_P1_PR_DIR
            gtfile=configs.H36M_P1_GT
        data = np.load(gtfile, allow_pickle=True)  
        J24_4d = data['S'] #(27558,24,4)
        self.J17_3d = self.root_centered(J24_4d)
        # self.bbox_centers = data['center'] #27588*2
        # self.scales = data['scale']*200 #27588
        # import ipdb; ipdb.set_trace()
        
        used_samples = []
        self.imgfiles = []
        self.prfiles = []
        for n in range(self.J17_3d.shape[0]):
            imgfile = f'{imgpath}/{n}.png'
            prfile = f'{prpath}/{n}.npz'
            if os.path.exists(imgfile) and os.path.exists(prfile):
                used_samples.append(n)
                self.imgfiles.append(imgfile)
                self.prfiles.append(prfile)

        self.J17_3d = self.J17_3d[used_samples]
        self.num_samples = len(used_samples)
        

    def __len__(self):
        return self.num_samples

    def root_centered(self, S):
        J24_to_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
        J24_to_J14 = [0,1,2,3,4,5,6,7,8,9,10,11,17,18]
        # joints_name = ("R_Ankle0", "R_Knee1", "R_Hip2", "L_Hip3", "L_Knee4", "L_Ankle5", "R_Wrist6",
        #         "R_Elbow7", "R_Shoulder8", "L_Shoulder9", "L_Elbow10", "L_Wrist11", "Thorax12",
        #         "Head13", "HeadTop14")
        # J24_to_J14 = config.H36M_TO_J14
        
        center = (S[:,2,:3] +  S[:,3,:3])/2 #between two hip points
        center = np.expand_dims(center, axis=1)
        S = S[:, J24_to_J17, :3]
        # S = S - center
        return S

    def __getitem__(self, index):
        #GT
        GT_j17_3d = self.J17_3d[index]
        #
        imgname = self.imgfiles[index]
        image = cv2.imread(imgname)
        data = np.load(self.prfiles[index],allow_pickle=True)
        iuv = data['iuv'] #(imgh, imgw, 3)
        j2d = data['j2d'] #(17,3)
        
        return {'image':image,
                'iuv': iuv,
                'j2d': j2d,
                'j17_3d': GT_j17_3d}

class TestPr_3DPW(Dataset):
    def __init__(self, imgpath=configs.D3PW_IMG_DIR, prpath=configs.D3PW_PR_DIR, gtfile=configs.D3PW_GT):
        data =  np.load(gtfile, allow_pickle=True)
        self.pose = data['pose']
        self.shape = data['shape']
        #
        used_samples = []
        self.imgfiles = []
        self.prfiles = []
        for n in range(self.pose.shape[0]):
            imgfile = f'{imgpath}/{n}.png'
            prfile = f'{prpath}/{n}.npz'
            if os.path.exists(imgfile) and os.path.exists(prfile):
                used_samples.append(n)
                self.imgfiles.append(imgfile)
                self.prfiles.append(prfile)

        self.pose = self.pose[used_samples]
        self.shape = self.shape[used_samples]
        self.num_samples = len(used_samples)
        # self.prep_wh = proxy_rep_input_wh
        # self.crop = crop
        # self.bbox_centers = data['center'] #35515*2
        # self.scales = data['scale']*200 #35515
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        #GT
        pose = self.pose[index]
        shape = self.shape[index]
        #
        imgname = self.imgfiles[index]
        image = cv2.imread(imgname)
        data = np.load(self.prfiles[index],allow_pickle=True)
        iuv = data['iuv'] #(imgh, imgw, 3)
        j2d = data['j2d'] #(17,3)
        
        return {'image':image,
                'iuv': iuv,
                'j2d': j2d,
                'pose': pose,
                'shape': shape}