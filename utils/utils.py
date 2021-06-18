import copy

import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def load_pretrained_transformer(transformer, path):
    state_dict = torch.load(path)['state_dict']

    pretrained_dict = {}
    for name in state_dict.keys():
        if name.startswith('transformer'):
            new_name = '.'.join(name.split('.')[1:])
            pretrained_dict[new_name] = state_dict[name]
    
    transformer.load_state_dict(pretrained_dict)
    return transformer

def load_pretrained_embedding(embedding, path):
    state_dict = torch.load(path)['state_dict']

    pretrained_dict = {}
    for name in state_dict.keys():
        if name.startswith('embedding'):
            new_name = '.'.join(name.split('.')[1:])
            pretrained_dict[new_name] = state_dict[name]
            
    embedding.load_state_dict(pretrained_dict)
    return embedding

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
def parse_arguments(parser):
    parser.add_argument('--experiment',
                        help="Experiment setup to be run. Choose either 'sup' for supervised or 'ssl' \
                            for semisupervised")
    parser.add_argument('--no_cat',
                        help="number of categorical variables in the dataset (including the cls column)")
    parser.add_argument('--no_num',
                        help="number of numerical variables in the dataset (including the cls column)")
    parser.add_argument('--pretrained_checkpoint',
                        help="full path to ssl pretrained checkpoint to be finetuned")
            
    # parser.add_argument('--checkpoint-file',
    #                     help="Checkpoint file .ckpt of phoneme pretrained model to use for finetuning")
    # parser.add_argument('--freeze-feature-extractor', dest='FREEZE_FEATURE_EXTRACTOR',
    #                     default=False, action='store_true', 
    #                     help="Whether to freexe the backbone or update it during finetuning")
    # parser.add_argument('--use-eng-asr', dest='USE_ENG_ASR',
    #                     default=False, action='store_true', 
    #                     help="Whether to use Eng-ASR as backbone during finetuning")
    # parser.add_argument('--use-random-model', dest='USE_RANDOM_MODEL', 
    #                     default=False, action='store_true',
    #                     help="use a pretrained model to initialize the weights")
    args = parser.parse_args()
    return args



    


    

