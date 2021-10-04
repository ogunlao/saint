from typing import Dict
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

from .dataset import generate_dataset

def generate_dataloader(train_bs: int, val_bs: int, 
                        num_workers: int, data_paths: dict, **kwargs,):
    train_dataset, val_dataset, test_dataset = generate_dataset(**data_paths)
    
    # create sampler
    train_weight = train_dataset.make_weights_for_imbalanced_classes()
    train_weight = torch.from_numpy(train_weight)
    train_sampler = WeightedRandomSampler(train_weight.type('torch.DoubleTensor'), 
                                        len(train_weight),
                                        replacement=False,)
    
    pin_memory = True if torch.cuda.is_available() else False
    
    # Generate dataloaders
    train_loader = DataLoader(train_dataset, batch_size=train_bs, 
                            sampler=train_sampler, num_workers=num_workers, 
                            pin_memory=pin_memory)
        
    validation_loader = DataLoader(val_dataset, batch_size=val_bs, 
                                num_workers=num_workers,  
                                pin_memory=pin_memory, shuffle=False,)
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=val_bs, 
                                num_workers=num_workers, 
                                pin_memory=pin_memory, shuffle=False,)
    
    return train_loader, validation_loader, test_loader
