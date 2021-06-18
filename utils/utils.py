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



    


    

