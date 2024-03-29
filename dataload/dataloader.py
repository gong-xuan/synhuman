from torch.utils.data import DataLoader
from .syn_data import SyntheticData
from .test_pr_data import TestPr_H36M, TestPr_3DPW, TestPr_MPI3DHP, TestPr_SSP3D
from .test_img_data import TestImg_H36M, TestImg_3DPW, TestImg_MPI3DHP, TestImg_SSP3D
import configs

def Build_Train_Dataloader(batch_size, num_workers=4, pin_memory = False, valdata=''):
   
    train_dataset = SyntheticData(configs.STRAP_TRAIN_PATH)
    if valdata=='h36m':
        val_dataset = TestPr_H36M()
    elif valdata=='3dpw':
        val_dataset = TestPr_3DPW()
    else:
        val_dataset = SyntheticData(configs.STRAP_VAL_PATH)
        
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers,
                                  pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                drop_last=True, num_workers=num_workers,
                                pin_memory=pin_memory)
   
    return train_dataloader, val_dataloader

def Build_Test_Dataloader(args, pr_path='', img_path=''):
    pr_exist = pr_path and img_path
    if args.data == 'h36mp1':
        dataset = TestPr_H36M(img_path, pr_path, protocal=1) if pr_exist else TestImg_H36M(protocal=1, crop=args.img_crop_scale)
        withshape = False
        metrics = ['mpjpe', 'mpjpe_pa']
    elif args.data == 'h36mp2':
        dataset = TestPr_H36M(img_path, pr_path, protocal=2) if pr_exist else TestImg_H36M(protocal=2, crop=args.img_crop_scale)
        withshape = False
        metrics = ['mpjpe', 'mpjpe_pa']
    elif args.data == 'ssp3d':
        dataset = TestPr_SSP3D(img_path, pr_path) if pr_exist else TestImg_SSP3D(crop=args.img_crop_scale)
        withshape = True
        metrics= ['mpjpe_pa', 'pve-ts_sc']
    elif args.data == '3dpw':
        dataset = TestPr_3DPW(img_path, pr_path) if pr_exist else TestImg_3DPW(crop=args.img_crop_scale)
        withshape = True
        metrics = ['mpjpe_pa','pve']#['mpjpe','mpjpe_sc', 'mpjpe_pa', 'pve']      
    elif args.data == 'mpi':
        dataset = TestPr_MPI3DHP(img_path, pr_path) if pr_exist else TestImg_MPI3DHP(crop=args.img_crop_scale)
        withshape = False
        metrics = ['mpjpe_pa', 'pck_pa']
    print(len(dataset))
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=args.shuffle,
                            drop_last=False, 
                            num_workers=args.num_workers, 
                            pin_memory=True)
    
    return dataloader, withshape, metrics

