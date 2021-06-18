import torch
import torch.nn as nn

import copy

from models import clones

class CategoricalEmbedding(nn.Module):
    """
    Embedding of catogrical features using NN embedding layer.
    N embeddings will be created for N categorical feature
    """
    def __init__(self, num_of_categories, embed_dim):
        super(CategoricalEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_of_categories,
            embed_dim
        )
    
    def forward(self, x):
        # x: bs 
        return self.embedding(x) # bs, embed_dim
  

class NumericalEmbedding(nn.Module):  
    """
    Embedding class of each Numerical feature using NN Linear layer of size (1 x embed_dim) and then Relu non-linearity. 
    N embeddings will be created for N numerical features
    """
    def __init__(self, embed_dim):
        super(NumericalEmbedding, self).__init__()
        self.linear = nn.Sequential(
                            nn.Linear(1, embed_dim), 
                            nn.ReLU())

    def forward(self, x):
        # x: bs
        x = x.unsqueeze(1) # bs,1
        return self.linear(x)  # bs, embed_dim
    
class Embedding(nn.Module):
    """
    Do the embedding of catogrical and numerical data
    """
    def __init__(self, embed_dim, no_num, cats):
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.cat_embedding = nn.ModuleList()
        for cat in cats:
            self.cat_embedding.append(
                nn.Embedding(cat, embed_dim)
            )
        self.fc = nn.Sequential(nn.Linear(out_features=embed_dim, in_features=1),
                                nn.ReLU()
                                )
        self.num_embedding = clones(self.fc, no_num)

        self.no_num = no_num
        self.no_cat = len(cats)

        
    def forward(self, x):
        bs = x.shape[0]

        output = []
    
        for i, layer in enumerate(self.cat_embedding):
            output.append(layer(x[:, i].long()))

        for i, layer in enumerate(self.num_embedding):
            output.append(layer(x[:, self.no_cat+i].unsqueeze(1).float()))

        data = torch.stack(output, dim=1) # bs, n, embed_size
        
        return data