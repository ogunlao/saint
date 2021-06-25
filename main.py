import argparse

from src.config import args as default_args
from models.model_generator import get_model
from src.train import setup_experiment
from src.dataloader import generate_dataloader
from utils.utils import parse_arguments, dotdict

    
def main(args):
    transformer, embedding = get_model(args.model, args.num_heads,
                                       args.embed_dim, args.num_layers, 
                                       args.d_ff, args.dropout, 
                                       args.dropout_ff, args.no_num, 
                                       args.no_cat, args.cats)
    
    train_loader, validation_loader, test_loader = generate_dataloader(
        args.experiment, args.seed, args)
    
    best_model_ckpt, _ = setup_experiment(transformer, embedding, 
                                            train_loader, validation_loader, test_loader,
                                            args.experiment, args.pretrained_checkpoint, args)
    print(f'Path to best model found during training: \n{best_model_ckpt}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser, default_args)
    args = dotdict(args)
    
    main(args)
