import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader

from config import args
from dataset import DatasetTabular


# Generate datasets

train_sup_dataset = DatasetTabular(train_df, train_y)
val_dataset = DatasetTabular(val_df, val_y)
test_dataset = DatasetTabular(test_df, test_y)

train_ssl_dataset = None
if args.num_supervised_train_data != 'all':
    train_ssl_dataset = DatasetTabular(train_ssl, train_ssl_y)

# create sample
train_sup_weight = train_sup_dataset.make_weights_for_balanced_classes()
train_sup_weight = torch.from_numpy(train_sup_weight)
train_sup_sampler = WeightedRandomSampler(train_sup_weight.type('torch.DoubleTensor'), len(train_sup_weight),
                                replacement=False,)

if args.num_supervised_train_data != 'all':
    train_ssl_weight = train_ssl_dataset.make_weights_for_balanced_classes()
    train_ssl_weight = torch.from_numpy(train_ssl_weight)
    train_ssl_sampler = WeightedRandomSampler(train_ssl_weight.type('torch.DoubleTensor'), len(train_ssl_weight),
                                replacement=True,)

pin_memory = True if torch.cuda.is_available() else False
    
# Generate dataloaders
train_sup_loader = DataLoader(train_sup_dataset, batch_size=args.batch_size, 
                          sampler=train_sup_sampler, num_workers=args.num_workers, 
                          pin_memory=pin_memory)

train_ssl_loader = None
if args.num_supervised_train_data != 'all':
    train_ssl_loader = DataLoader(train_ssl_dataset, batch_size=args.batch_size, 
                            sampler=train_ssl_sampler, num_workers=args.num_workers, 
                            pin_memory=pin_memory, shuffle=True,)
    
validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               num_workers= args.num_workers,  
                               shuffle=False,)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                         num_workers=args.num_workers, 
                         pin_memory=pin_memory, shuffle=False,)