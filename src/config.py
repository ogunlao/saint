from argparse import Namespace
import multiprocessing
import torch


args = Namespace(
    task = 'classification', # {'classification', 'regression'}
    #parameters for the model 
    num_output = 1, # {1 for binary, > 1 for multiclass}
    num_layers = 6,
    num_heads =  8,
    dropout = 0.1,
    embed_dim = 32,
    d_ff = 32,
    cls_token_idx = 0,

    # parameters for cutmix
    prob_cutmix = 0.3,

    # parameters for mixup
    alpha = 0.2, # [0.1, 0.4]
    lambda_pt = 10,

    # parameter for contrastive loss
    temperature = 0.7,
    proj_head_dim = 128,

    # parameters for the dataset
    data_folder = 'data',
    train_split = 0.65,
    validation_split = 0.15,
    test_split = 0.20,
    batch_size = 32, # [32, 256]
    num_workers = 8,
    num_supervised_train_data = 'all', # {'all', 50, 200, 500}

    # parameters for training
    beta_1 = 0.9,
    beta_2 = 0.99,
    learning_rate = 0.0001,
    weight_decay = 0.01,
    optim = 'adamw',
    freeze_encoder = True, # freeze transformer layer

    num_epochs = 200, # default is 100
    no_of_gpus = 4,
    seed = 1234, # 4 different seeds used
    resume_checkpoint = None,
    monitor = 'val_loss',
)

args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()