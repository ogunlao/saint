from argparse import Namespace
import multiprocessing
import torch

args = Namespace(
    # experiment parameters
    experiment = 'sup', # {'sup', 'ssl'}
    task = 'classification', # {'classification', 'regression'}
    model = 'saint',
    pretrained_checkpoint = None, # '/home/ola/Projects/saint/checkpoints/lightning_logs/version_0/checkpoints/epoch=0-step=1.ckpt',
    
    # path to csv files for training either ssl or sup
    train_csv_path = '/home/ola/Projects/saint/data/train.csv',
    train_y_csv_path = '/home/ola/Projects/saint/data/train_y.csv',
    val_csv_path = '/home/ola/Projects/saint/data/val.csv',
    val_y_csv_path = '/home/ola/Projects/saint/data/val_y.csv',
    test_csv_path = '/home/ola/Projects/saint/data/test.csv',
    test_y_csv_path = '/home/ola/Projects/saint/data/test_y.csv',
    
    #parameters for the model 
    num_output = 1, # {1 for binary, > 1 for multiclass}
    num_layers = 6,
    num_heads = 8,
    dropout = 0.1,
    dropout_ff = 0.1, # dropout for feedforward layers, 0.1 used for saint_s and 0.8 for saint and saint_i variants
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
    no_cat = 10,
    no_num = 7,
    cats = [1, 12, 3, 4, 2, 2, 2, 3, 12, 4],
    batch_size = 32, # [32, 256]
    num_workers = 8,
    
    # parameters for preprocessing dataset
    data_folder = 'data',
    train_split = 0.65,
    validation_split = 0.15,
    test_split = 0.20,
    num_supervised_train_data = 'all', # {'all', 50, 200, 500}

    # parameters for training
    beta_1 = 0.9,
    beta_2 = 0.99,
    learning_rate = 0.0001,
    weight_decay = 0.01,
    optim = 'adamw',
    freeze_encoder = True, # freeze transformer layer

    num_epochs = 1, # default is 100
    no_of_gpus = 4,
    seed = 1234, # 4 different seeds used
    resume_checkpoint = None,
    monitor = 'val_loss', # {val_loss, val_auroc_epoch}
    monitor_mode = 'min' # {max, min}
)

args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()