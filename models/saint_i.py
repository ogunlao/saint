"""The saint intersample code follows the pseudocode given in the \
    Saint Paper by Gowthami Somepalli et al. See https://arxiv.org/abs/2106.01342"""

import torch.nn as nn

from .transformer import attention, clones
from .transformer import PositionwiseFeedForward, EncoderLayer, Encoder

def intersample(query , key , value,dropout=None):
    "Calculate the intersample of a given query batch" 
    #x , bs , n , d 
    b, h, n , d = query.shape
    #print(query.shape,key.shape, value.shape )
    query , key , value = query.reshape(1, b, h, n*d), \
                            key.reshape(1, b, h, n*d), \
                                value.reshape(1, b, h, n*d)

    output, _ = attention(query, key ,value)  #1 , b, n*d
    output = output.squeeze(0) #b, n*d
    output = output.reshape(b, h, n, d) #b,n,d

    return output

class MultiHeadedIntersampleAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedIntersampleAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
     
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
     
    def forward(self, query, key, value):
        "Implements Figure 2"
       
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x = intersample(query, key, value, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) # bs , n , d_model
        return self.linears[-1](x)  # bs , n , d_model
    
    
def make_saint_i(num_heads, embed_dim, num_layers, d_ff, dropout, dropout_ff=0.8):
    """
    Creates the Saint-i model by stacking  intersample attention  and 
    feed forward into the encoder layer, the encoder layer is then stacked 
    with the embedding layer in the transformer object

    -----------
    Parameters
    num_heads: (int) number of attention heads 
    embed_dim: (int) size of embedding vector 
    num_layers : (int) numbe of self attention laters
    d_ff: (int)
    dropout: (float) How much activations to drop
    dropout_ff: (float) How much activations to drop in feedforward layers. Defaults to 0.8 
    """

    feed_forward = PositionwiseFeedForward(d_model=embed_dim, d_ff=d_ff, dropout=dropout_ff)
    self_attn = MultiHeadedIntersampleAttention(num_heads, d_model=embed_dim, dropout=dropout)
    

    layer = EncoderLayer(embed_dim, self_attn, feed_forward, dropout)
    encoder = Encoder(layer, num_layers)
    
    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return encoder
