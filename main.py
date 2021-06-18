from src.config import args
from model.model_generator import get_model
from train import setup_experiment
from dataloader import generate_dataloader

def main(args):
    # get model
    transformer, embedding = get_model(args.model, args)
    
    # get dataloaders
    train_loader, validation_loader, test_loader, train_ssl_loader = generate_dataloader(
        args.num_supervised_train_data, args.experiment, args.seed)
    
    pretrained_path = None
    best_model_ckpt, _ = setup_experiment(transformer, embedding, 
                                            train_loader, validation_loader, test_loader,
                                            args.experiment, pretrained_path, args)
    print(f'Best model seen during training: {best_model_ckpt}')
    

if __name__ == "__main__":
    main(args)