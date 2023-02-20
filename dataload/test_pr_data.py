import os, cv2
import numpy as np
from torch.utils.data import Dataset
import configs

def filter_orgdata_with_pr(orgfile_list, imgpath, prpath):
    used_idx = []
    imgfiles = []
    prfiles = []
    for n, orgfile in enumerate(orgfile_list):
        savename = orgfile.split('/')[-1][:-4]
        imgfile = f'{imgpath}/{savename}.png'
        prfile = f'{prpath}/{savename}.npz'
        if os.path.exists(imgfile) and os.path.exists(prfile):
            used_idx.append(n)
            imgfiles.append(imgfile)
            prfiles.append(prfile)
    return used_idx, imgfiles, prfiles

class TestPr_H36M(Dataset):
    def __init__(self, imgpath, prpath, protocal):
        if protocal==2: ##Skip: [8394]
            gtfile=configs.H36M_P2_GT
        elif protocal==1:       
            gtfile=configs.H36M_P1_GT
        
        data = np.load(gtfile, allow_pickle=True)  
        J24_4d = data['S'] #(27558,24,4)
        self.J17_3d = self.root_centered(J24_4d)

        imgfiles = data['imgname'].tolist()
        # self.bbox_centers = data['center'] #27588*2
        # self.scales = data['scale']*200 #27588
        
        used_idx, self.imgfiles, self.prfiles = filter_orgdata_with_pr(imgfiles, imgpath, prpath)

        self.J17_3d = self.J17_3d[used_idx]
        self.num_samples = len(used_idx)
        

    def __len__(self):
        return self.num_samples

    def root_centered(self, S):
        J24_to_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
        
        # joints_name = ("R_Ankle0", "R_Knee1", "R_Hip2", "L_Hip3", "L_Knee4", "L_Ankle5", "R_Wrist6",
        #         "R_Elbow7", "R_Shoulder8", "L_Shoulder9", "L_Elbow10", "L_Wrist11", "Thorax12",
        #         "Head13", "HeadTop14")
        # J24_to_J14 = config.H36M_TO_J14
        # J24_to_J14 = [0,1,2,3,4,5,6,7,8,9,10,11,17,18]
        
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


class TestPr_MPI3DHP(TestPr_H36M):
    def __init__(self,imgpath, prpath, gtfile=configs.MPI3DHP_GT):
        data = np.load(gtfile, allow_pickle=True)  
        J24_4d = data['S'] #(27558,24,4)
        self.J14_3d = self.root_centered(J24_4d)
        imgfiles =data['imgname'].tolist()
        # self.bbox_centers = data['center'] #27588*2
        # self.scales = data['scale']*200 #27588
        # import ipdb; ipdb.set_trace() 
        
        used_idx, self.imgfiles, self.prfiles = filter_orgdata_with_pr(imgfiles, imgpath, prpath)


        self.J14_3d = self.J14_3d[used_idx]
        self.num_samples = len(used_idx)


class TestPr_3DPW(Dataset):
    def __init__(self, imgpath, prpath, gtfile=configs.D3PW_GT):
        data =  np.load(gtfile, allow_pickle=True)
        self.pose = data['pose']
        self.shape = data['shape']
        
        imgfiles = data['imgname'].tolist()
        used_idx, self.imgfiles, self.prfiles = filter_orgdata_with_pr(imgfiles, imgpath, prpath)

        self.pose = self.pose[used_idx]
        self.shape = self.shape[used_idx]
        self.num_samples = len(used_idx)
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
        data = np.load(self.prfiles[index], allow_pickle=True)
        iuv = data['iuv'] #(imgh, imgw, 3)
        j2d = data['j2d'] #(17,3)
        
        return {'image':image,
                'iuv': iuv,
                'j2d': j2d,
                'pose': pose,
                'shape': shape}


class TestPr_SSP3D(TestPr_3DPW):
    def __init__(self, imgpath, prpath, gtfile=configs.SSP3D_GT, use_pred=True):
        data =  np.load(gtfile, allow_pickle=True)
        self.use_pred =use_pred
        self.pose = data['poses']
        self.shape = data['shapes']
        self.gender = data['genders']
        self.joints2D = data['joints2D']
        #
        imgfiles = data['fnames'].tolist()
        used_idx, self.imgfiles, self.prfiles = filter_orgdata_with_pr(imgfiles, imgpath, prpath)

        self.pose = self.pose[used_idx]
        self.shape = self.shape[used_idx]
        self.num_samples = len(used_idx)
    
    def __getitem__(self, index):
        #GT
        pose = self.pose[index]
        shape = self.shape[index]
        gender = self.gender[index]
        #
        imgname = self.imgfiles[index]
        image = cv2.imread(imgname)
        data = np.load(self.prfiles[index],allow_pickle=True)
        iuv = data['iuv'] #(imgh, imgw, 3)
        if self.use_pred:
            j2d = data['j2d'] #(17,3)
        else:
            j2d = self.joints2D[index][:,:2]
        return {'image':image,
                'iuv': iuv,
                'j2d': j2d,
                'pose': pose,
                'shape': shape,
                'gender':gender}