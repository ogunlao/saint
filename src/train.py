import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.trainer import SaintSemiSupLightningModule, SaintSupLightningModule
from src.config import args
import copy

def setup_experiment(transformer, embedding, 
                     train_loader, validation_loader, test_loader,
                     experiment, pretrained_checkpoint, args):
    
    seed_everything(args.seed)

    if experiment == 'ssl':
        model = SaintSemiSupLightningModule(transformer, embedding,
                                            args.optim, args.learning_rate,
                                            args.weight_decay, args.prob_cutmix,
                                            args.alpha, args.lambda_pt,
                                            args.embed_dim, args.proj_head_dim,
                                            args.no_num, args.no_cat,
                                            args.cats, args.temperature,
                                            args.task,)
    elif experiment == 'sup':
        fc = nn.Linear(args.embed_dim, args.num_output)
        model = SaintSupLightningModule(transformer, embedding,
                                        fc, args.optim,
                                        args.learning_rate,
                                        args.weight_decay,
                                        args.task, args.num_output,
                                        args.cls_token_idx,
                                        args.freeze_encoder,
                                        args.metric,
                                        )
        
        if pretrained_checkpoint is not None:
            print(f'Initializing sup task using pretrained model:\n{pretrained_checkpoint}')
            ssl_lm_params = dict(transformer=transformer, embedding=embedding,
                             optim=args.optim, lr=args.learning_rate, 
                             weight_decay=args.weight_decay, prob_cutmix=args.prob_cutmix, 
                             alpha=args.alpha, lambda_pt=args.lambda_pt,
                             embed_dim=args.embed_dim, proj_head_dim=args.proj_head_dim,
                             no_num=args.no_num, no_cat=args.no_cat, cats=args.cats,
                             temperature=args.temperature, task=args.task,)
            
            model_ssl = SaintSemiSupLightningModule(**ssl_lm_params)
            model_ssl.load_from_checkpoint(pretrained_checkpoint, **ssl_lm_params)
            
            model.transformer = copy.deepcopy(model_ssl.transformer)
            model.embedding = copy.deepcopy(model_ssl.embedding)
            
    else:
        print('Unknown experiment type. Select either "sup" or "ssl"')
        exit()
    
    checkpoint_callback = ModelCheckpoint(monitor=args.monitor,
                                          mode=args.monitor_mode)

    # training
    trainer = pl.Trainer(gpus=args.no_of_gpus,
                         deterministic=True,
                         callbacks=[checkpoint_callback],
                         max_epochs=args.num_epochs,
                         default_root_dir='checkpoints',

                         # for sanity checks
                         # overfit_batches=1,
                         # num_sanity_val_steps=0,
                         # resume_from_checkpoint = args.resume_checkpoint,
                         )

    trainer.fit(model, train_loader, validation_loader,)

    if experiment == 'sup' and test_loader is not None:
        trainer.test(ckpt_path='best',
                    test_dataloaders=test_loader,
                    )

    best_model_ckpt = checkpoint_callback.best_model_path
    best_model_score = checkpoint_callback.best_model_score

    return best_model_ckpt, best_model_score
