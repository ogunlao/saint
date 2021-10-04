from .saint import make_saint
from .saint_i import make_saint_i
from .saint_s import make_saint_s

from .embedding import Embedding

model_type = dict(saint = make_saint,
                saint_i = make_saint_i,
                saint_s = make_saint_s
                )

def get_model(model_name, num_heads, embed_dim, num_layers, 
             d_ff, dropout, dropout_ff, no_num, 
             no_cat, cats, *args, **kwargs):
    model_fn = model_type[model_name]
    encoder = model_fn(num_heads, embed_dim, num_layers, 
             d_ff, dropout, dropout_ff)
    embedding = Embedding(embed_dim, no_num, 
                          no_cat, cats)
    
    return encoder, embedding
    
    