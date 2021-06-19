import torch
import torch.nn as nn

from .transformer import PositionwiseFeedForward, MultiHeadedAttention
from .transformer import EncoderLayer, Encoder

def make_saint_s(num_heads, embed_dim, num_layers, d_ff, dropout, dropout_ff=0.1):
    """
    Make the Saint-t model by stacking  mutlihead attention 
    and feed forward into the encoder layer

    -----------
    Paramaters
    -----------
    num_heads: (int) number of attention heads 
    embed_dim: (int) size of embedding vector 
    num_layers : (int) numbe of self attention laters
    d_ff: (int)
    dropout: (float) How much activations to drop
    dropout_ff: (float) How much activations to drop in feedforward layers. Defaults to 0.1
    """

    feed_forward = PositionwiseFeedForward(d_model=embed_dim, 
                                           d_ff=d_ff, 
                                           dropout=dropout_ff)
    self_attn = MultiHeadedAttention(num_heads, 
                                     d_model=embed_dim, 
                                     dropout=dropout)
    
    layer = EncoderLayer(embed_dim, self_attn, 
                         feed_forward, dropout)
    encoder = Encoder(layer, num_layers)

    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return encoder