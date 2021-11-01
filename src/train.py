import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.trainer import SaintSemiSupLightningModule, SaintSupLightningModule
import copy

def setup_experiment(model_dict: dict, 
                     dataloaders: dict,
                     args):

    experiment = args.experiment.experiment
    mode = {'supervised': SaintSupLightningModule, 
            'self-supervised': SaintSemiSupLightningModule}
    
    model_init = mode.get(experiment, None)
    if model_init is None:
        raise NameError('Unknown experiment type. Select either "supervised" \
                        or "self-supervised" in config default')
    
    fc = nn.Linear(args.transformer.embed_dim, args.experiment.num_output) if experiment=='supervised' else None
    model_dict['fc'] = fc
    
    params = dict(**model_dict, **args.optimizer, **args.augmentation, 
                       **args.transformer, **args.data.data_stats, **args.experiment,)
    
    model = model_init(**params)
    
    if args.experiment.pretrained_checkpoint is not None:
        pt_ckpt = args.experiment.pretrained_checkpoint
        print(f'Initializing supervised task using pretrained model:\n{pt_ckpt}')
        
        pt_model_ssl = SaintSemiSupLightningModule(**params)
        pt_model_ssl.load_from_checkpoint(pt_ckpt, **params)
        
        # copy weights from pretrained
        model.transformer = copy.deepcopy(pt_model_ssl.transformer)
        model.embedding = copy.deepcopy(pt_model_ssl.embedding)
    
    trainer, checkpoint_callback = setup_trainer(args.trainer, args.callback)

    trainer.fit(model, dataloaders['train_loader'], dataloaders['validation_loader'])

    if experiment == 'supervised' and dataloaders['test_loader'] is not None:
        trainer.test(ckpt_path='best',
                    test_dataloaders=dataloaders['test_loader'],
                    )

    best_model_ckpt = checkpoint_callback.best_model_path
    best_model_score = checkpoint_callback.best_model_score

    return best_model_ckpt, best_model_score
    

def setup_trainer(trainer_args: dict, cb_args: dict, **kwargs):
    checkpoint_callback = ModelCheckpoint(**cb_args)

    trainer = pl.Trainer(**trainer_args,
                         callbacks=[checkpoint_callback],
                         )
    
    return trainer, checkpoint_callback
