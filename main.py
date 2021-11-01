import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from models.model_generator import get_model
from src.train import setup_experiment
from src.dataloader import generate_dataloader


@hydra.main(config_path="configs", config_name="config")
def main(args: DictConfig) -> None:
    if args.print_config is True:
        print(args)
    seed_everything(args.seed)
    
    transformer, embedding = get_model(args.experiment.model, 
                                       **args.transformer, 
                                       **args.data.data_stats,)
    
    train_loader, validation_loader, test_loader = generate_dataloader(
        data_paths=args.data.data_paths, **args.dataloader, )
    
    model_dict = dict(transformer=transformer, 
                      embedding=embedding)
    dataloaders = dict(train_loader=train_loader,
                       validation_loader=validation_loader,
                       test_loader=test_loader)
    best_model_ckpt, _ = setup_experiment(model_dict, dataloaders, args,)
    print(f'Path to best model found during training: \n{best_model_ckpt}')
    

if __name__ == "__main__":   
    main()
