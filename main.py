import argparse
from collections import ChainMap 

from src.config import args as default_args
from models.model_generator import get_model
from train import setup_experiment
from src.dataloader import generate_dataloader
from utils.utils import parse_arguments, dotdict

    
def main(args):
    transformer, embedding = get_model(args.model, args)
    
    train_loader, validation_loader, test_loader, train_ssl_loader = generate_dataloader(
        args.num_supervised_train_data, args.experiment, args.seed, args)
    
    best_model_ckpt, _ = setup_experiment(transformer, embedding, 
                                            train_loader, validation_loader, test_loader,
                                            args.experiment, args.pretrained_path, args)
    print(f'Path to best model found during training: \n{best_model_ckpt}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    args_col = ChainMap(vars(args), vars(default_args))
    args_col = dotdict(args_col)
    
    main(args_col)