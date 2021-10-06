import copy

import torch
import torch.nn as nn

from torchmetrics import AUROC, Accuracy
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

class Metric:
    "Metrics dispatcher. Adapted from answer at https://stackoverflow.com/a/58923974"
    def __init__(self, num_classes):
        self.num_classes=num_classes

    def get_metric(self, metric='acc'):
        """Dispatch metric with method"""
        
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, metric, lambda: "Metric not implemented yet")
        
        return method()

    def auroc(self):
        return AUROC(num_classes=self.num_classes,)

    def acc(self):
        return  Accuracy(num_classes=self.num_classes)



    


    

