import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau



seed_everything(args.seed, workers=True)

# create embeddings


embedding = Embedding(args.embed_dim, args.no_num, args.cats)
transformer_t = make_transformer()


# load transformer pretrained weight
pretrained_t_state_dict = load_transformer_state_dict(best_ssl_model_ckpt)
transformer_t.load_state_dict(pretrained_t_state_dict,)

# load embedding pretrained weight
pretrained_embedding_state_dict = load_embedding_state_dict(best_ssl_model_ckpt)
embedding.load_state_dict(pretrained_embedding_state_dict)


# Define fc layer
fc = nn.Linear(args.embed_dim, args.num_output)

lt_model = SaintSupLightningModule(transformer_t, embedding, fc, args.optim, 
                            args.learning_rate,
                            args.weight_decay,
                            args.task, args.num_output, 
                            args.cls_token_idx, 
                            args.freeze_encoder)
    

supervised_checkpoint_callback = ModelCheckpoint(monitor=args.monitor,
                                    mode='min')

# training

supervised_trainer = pl.Trainer(gpus=args.no_of_gpus,
                    deterministic=True,
                    callbacks=[supervised_checkpoint_callback],
                    max_epochs=args.num_epochs,
                    default_root_dir = 'checkpoints',
                    
                    # for sanity checks
                    # overfit_batches=1, 
                    # num_sanity_val_steps=0,
                    
                    # resume_from_checkpoint = args.resume_checkpoint,
                    )

supervised_trainer.fit(lt_model, train_sup_loader, validation_loader,)


# evaluate on test set
supervised_trainer.test(ckpt_path='best',
                test_dataloaders=test_loader,
                )