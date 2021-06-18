import torch
import torch.nn as nn

from .transformer import PositionwiseFeedForward, MultiHeadedAttention
from .transformer import EncoderLayer, Encoder

def make_saint_s(num_heads, embed_dim, num_layers, d_ff, dropout):
    """
    Make the Saint-t model by stacking  mutlihead attention 
    and feed forward into the encoder layer

    -----------
    Parameters
    num_len: int of number of numerical features 
    cat_lest: List of number of catogeries 
    """

    feed_forward = PositionwiseFeedForward(d_model=embed_dim, d_ff=d_ff)
    self_attn = MultiHeadedAttention(num_heads,d_model=embed_dim)
    
    layer = EncoderLayer(embed_dim, self_attn, feed_forward, dropout)
    encoder = Encoder(layer, num_layers)
    # model = Transformer(encoder)

    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return encoder