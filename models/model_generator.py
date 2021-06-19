from .saint import make_saint
from .saint_i import make_saint_i
from .saint_s import make_saint_s

from .embedding import Embedding

model_type = dict(saint = make_saint,
                saint_i = make_saint_i,
                saint_s = make_saint_s
                )

def get_model(model_name, args):
    model_fn = model_type[model_name]
    encoder = model_fn(args.num_heads, args.embed_dim, args.num_layers, 
             args.d_ff, args.dropout, args.dropout_ff)
    embedding = Embedding(args.embed_dim, args.no_num, 
                          args.no_cat, args.cats)
    
    return encoder, embedding
    
    