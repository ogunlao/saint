"""The saint transformer code follows the pseudocode give in the \
    Saint Paper by Gowthami Somepalli et al. See https://arxiv.org/abs/2106.01342"""

import torch.nn as nn

from .transformer import PositionwiseFeedForward, MultiHeadedAttention
from .transformer import EncoderLayer, Encoder

from .saint_i import MultiHeadedIntersampleAttention


class SaintLayer(nn.Module):
    """Saint layer stacks the self attention and the intersample attention together"""
    def __init__(self, msa, misa, size):
        """
        Args:
            msa (nn.Module): Encoder layer of the self attention 
            misa (nn.Module): Encoder layer of intersample attention 
            size (int): Number of features, this is required by LayerNorm
        """
        super(SaintLayer, self).__init__()
        self.msa = msa              # multi-head attention
        self.misa = misa            # multi-head interasample attention
        self.size = size

    def forward(self, x):
        return self.misa(self.msa(x))


def make_saint(num_heads, embed_dim, num_layers, d_ff, dropout, dropout_ff=0.8):
    """Make the Saint model by stacking  the Saint layer  into the encoder layer, \
    the encoder layer is then stacked with the embedding layer in the transformer object
    
    Args:
        num_heads (int): number of attention heads 
        embed_dim (int): size of embedding vector 
        num_layers (int): numbe of self attention laters
        d_ff (int):
        dropout (float): How much activations to drop
        dropout_ff (float, optional): How much activations to drop in feed forward layers. Defaults to 0.8.
    
    Returns:
        nn.Module: transformer encoder
    """
    feed_forward = PositionwiseFeedForward(d_model=embed_dim, d_ff=d_ff, dropout=dropout_ff)
    self_attn = MultiHeadedAttention(num_heads, d_model=embed_dim)
    msa = EncoderLayer(embed_dim, self_attn, feed_forward, dropout)

    self_inter_attn = MultiHeadedIntersampleAttention(
        num_heads, d_model=embed_dim)
    misa = EncoderLayer(embed_dim, self_inter_attn, feed_forward, dropout)

    layer = SaintLayer(msa, misa, size=embed_dim)

    encoder = Encoder(layer, num_layers)
    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return encoder
