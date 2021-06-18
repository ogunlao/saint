import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.trainer import SaintSemiSupLightningModule, SaintSupLightningModule
from utils.utils import load_pretrained_transformer, load_pretrained_embedding
from src.config import args

def setup_experiment(transformer, embedding, 
                     train_loader, validation_loader, test_loader,
                     experiment, pretrained_path, args):
    
    seed_everything(args.seed, workers=True)

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
        if pretrained_path is not None:
            transformer = load_pretrained_transformer(
                                            transformer, pretrained_path)
            embedding = load_pretrained_embedding(embedding, pretrained_path)

        fc = nn.Linear(args.embed_dim, args.num_output)
        model = SaintSupLightningModule(transformer, embedding,
                                        fc, args.optim,
                                        args.learning_rate,
                                        args.weight_decay,
                                        args.task, args.num_output,
                                        args.cls_token_idx,
                                        args.freeze_encoder)
    else:
        print('Unknown experiment type. Select either "sup" or "ssl"')
        
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          mode='min')

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

    if experiment == 'ssl' and test_loader is not None:
        trainer.test(ckpt_path='best',
                    test_dataloaders=test_loader,
                    )

    best_model_ckpt = checkpoint_callback.best_model_path
    best_model_score = checkpoint_callback.best_model_score

    return best_model_ckpt, best_model_score
