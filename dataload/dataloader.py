from torch.utils.data import DataLoader
from .syn_data import SyntheticData
from .test_pr_data import TestPr_H36M, TestPr_3DPW
import configs

def Build_TrainDataloader(batch_size, num_workers=4, pin_memory = False, valdata=''):
   
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