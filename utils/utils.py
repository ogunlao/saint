import copy

import torch
import torch.nn as nn

import argparse
from collections import ChainMap 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def load_pretrained_model(model, path, model_name='transformer'):
    print(f'Loading pretrained {model_name}...')
    state_dict = torch.load(path)['state_dict']

    pretrained_dict = {}
    for name in state_dict.keys():
        if name.startswith(model_name):
            new_name = '.'.join(name.split('.')[1:])
            pretrained_dict[new_name] = state_dict[name]
    
    model.load_state_dict(pretrained_dict)
    return model


class dotdict(dict):
    """dot.notation access to dictionary attributes
    Source: How to use a dot “.” to access members of dictionary? \
    https://stackoverflow.com/a/23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
def parse_arguments(parser, default_args):
    parser.add_argument('--experiment', dest='experiment', 
                        default=default_args.experiment, type=str,
                        help="Experiment setup to be run. Choose either 'sup' for supervised or 'ssl' \
                            for semisupervised")
    parser.add_argument('--no_cat', dest='no_cat', type=int, 
                        default=default_args.no_cat,
                        help="number of categorical variables in the dataset (including the cls column)")
    parser.add_argument('--no_num', dest='no_num', type=int, 
                        default=default_args.no_num,
                        help="number of numerical variables in the dataset (including the cls column)")
    parser.add_argument('--cats', 
                        default=default_args.cats, type=list,
                        help="no. of categories of each categorical feature as a list")
    parser.add_argument('--pretrained_checkpoint', type=str,
                        default=default_args.pretrained_checkpoint,
                        help="full path to ssl pretrained checkpoint to be finetuned")
    parser.add_argument('--model', default=default_args.model, type=str,
                        help="Select saint model to initialize", 
                        choices=['saint', 'saint_s', 'saint_i'], 
                        )
    args = parser.parse_args()
    args_col = ChainMap(vars(args), vars(default_args))    
    
    return args_col



    


    

