"""Create semi-supervised training and testing module"""
from src.config import args

seed_everything(args.seed, workers=True)


embedding = Embedding(args.embed_dim, args.no_num, args.cats)
transformer_t = make_transformer()

ssl_model = SaintSemiSupLightningModule(transformer_t, embedding, args.optim, 
                            args.learning_rate, args.weight_decay, 
                            args.prob_cutmix, args.alpha, args.lambda_pt,
                            args.embed_dim, args.proj_head_dim, 
                            args.no_num, args.no_cat, args.cats, args.temperature,
                            args.task,)
    

ssl_checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                    mode='min')
    
# training
ssl_trainer = pl.Trainer(gpus=args.no_of_gpus,
                    deterministic=True,
                    callbacks=[ssl_checkpoint_callback],
                    max_epochs=args.num_epochs,
                    default_root_dir = 'checkpoints',
                    
                    # for sanity checks
                    # overfit_batches=1, 
                    # num_sanity_val_steps=0,
                    # resume_from_checkpoint = args.resume_checkpoint,
                    )


ssl_trainer.fit(ssl_model, train_sup_loader, validation_loader,)


best_ssl_model_ckpt = ssl_checkpoint_callback.best_model_path
best_ssl_model_score = ssl_checkpoint_callback.best_model_score

        