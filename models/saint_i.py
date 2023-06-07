"""The saint intersample code follows the pseudocode given in the \
    Saint Paper by Gowthami Somepalli et al. See https://arxiv.org/abs/2106.01342"""

import torch.nn as nn

from .transformer import attention, clones
from .transformer import PositionwiseFeedForward, EncoderLayer, Encoder

class MultiHeadedIntersampleAttention(nn.Module):
    '''
     Wrapper class for MHA which calculate attention over samples rather than features
    '''
    
    def __init__(self, *args, **kwargs):
        '''
         Arguments are passed to MHA class
        '''
        
        # initalise MHA attention layer
        super().__init__(*args, **kwargs)
     
        
    # Overwite forward method to transpose
    def forward(self, query, key, value, **kwargs):
        '''
         Requires query, key, value vectors of size batcn x d_model, transpoes and calucaltes attention across samples, transposes back and returns
         kwargs are passed directly to nn.MultiheadAttention
        '''
        
        # transpose q, k, v to shape d_model x batch
        query = query.transpose(0,1)
        key = key.transpose(0,1)
        value = value.transpose(0,1)
        
        output, attn_output_weights = super().forward(query, key, value, **kwargs)  # call forward function for MHA
        
        return output.transpose(0,1)  # return 

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
