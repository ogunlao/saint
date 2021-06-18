import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics import AUROC, Accuracy

from utils.augment import CutMix, Mixup
from utils.loss import ConstrastiveLoss, DenoisingLoss

class SaintSupLightningModule(pl.LightningModule):
    def __init__(self, transformer, embedding, fc, optim, lr,
                 weight_decay, task, num_output, cls_token_idx, freeze_encoder=False):
        super().__init__()
        self.transformer = transformer
        self.embedding = embedding
        if freeze_encoder:
            self.transformer.eval()
            # freeze params
            for param in self.transformer.parameters():
                param.requires_grad = False
            self.embedding.eval()
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.fc = fc
        self.lr = lr
        self.optim = optim # 'adamw'
        self.weight_decay = weight_decay
        self.num_classes = None if num_output == 1 else num_output
        self.task = task # {'classification', 'regression'}
        self.cls_token_idx = cls_token_idx
        self.setup_criterion()

        self.train_auroc = AUROC(num_classes=self.num_classes,)
        self.valid_auroc = AUROC(num_classes=self.num_classes,)
        self.test_auroc = AUROC(num_classes=self.num_classes,)

        self.train_acc = Accuracy(num_classes=self.num_classes)
        self.valid_acc = Accuracy(num_classes=self.num_classes)
        self.test_acc = Accuracy(num_classes=self.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # cls_vec  BS x embed_dim
  
        out = self.fc(x[:, self.cls_token_idx, :].squeeze())
        return out
    
    def _shared_step(self, batch, auroc_fn, accuracy_fn):
        x, targets = batch
        targets = targets.squeeze()
        
        x = self.embedding(x)
        x = self.transformer(x)
        # cls_vec  BS x embed_dim x[:, :, 0].squeeze()
        outputs = self.fc(x[:, self.cls_token_idx, :]).squeeze()
        
        loss = self.criterion(outputs, targets.squeeze().float())
        preds = torch.sigmoid(outputs)

        auroc_fn.update(preds, targets)
        accuracy_fn.update(preds, targets)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.train_auroc, self.train_acc)
        
        # log the outputs!
        self.log('train_loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        self.log(f'train_accuracy_epoch', self.train_acc.compute(), prog_bar=True,)
        self.log(f'train_auroc_epoch', self.train_auroc.compute(), prog_bar=True,)

        # reset after each epoch # May not be necessary, boiler plate should do it itself
        self.train_acc.reset()
        self.train_auroc.reset()

    def validation_step(self, batch, batch_idx):
        val_loss = self._shared_step(batch, self.valid_auroc, self.valid_acc)
        
        # log the outputs!
        self.log(f'val_loss', val_loss, on_step=False, 
                 on_epoch=True, prog_bar=True, logger=True)
        
    def validation_epoch_end(self, validation_step_outputs):
        self.log(f'valid_accuracy_epoch', self.valid_acc.compute(), prog_bar=True,)
        self.log(f'valid_auroc_epoch', self.valid_auroc.compute(), prog_bar=True,)

        # reset after each epoch
        self.valid_acc.reset()
        self.valid_auroc.reset()

    def test_step(self, batch, batch_idx):
        test_loss = self._shared_step(batch, self.test_auroc, self.test_acc)
        
        # log the outputs!
        self.log('test_loss', test_loss, on_step=False, 
                 on_epoch=True, prog_bar=True, logger=True)
    
    def test_epoch_end(self, test_outs):
        self.log(f'test_accuracy_best_epoch', self.test_acc.compute())
        self.log(f'test_auroc_best_epoch', self.test_auroc.compute())

    def configure_optimizers(self):
        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr,
                                        betas=(0.9, 0.999),
                                        weight_decay=self.weight_decay)
        else: # put as dummy optim
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': ReduceLROnPlateau(optimizer, 
                                                   patience=10, 
                                                   min_lr=1e-8, 
                                                   factor=0.5,
                                                   verbose=True),
                    'monitor': 'val_loss',
                }
            }

    def setup_criterion(self):   
        if self.task == 'classification':
            if self.num_classes is None:      
                self.criterion = nn.BCEWithLogitsLoss()
            else: # multiclass
                self.criterion = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            self.criterion = nn.MSELoss()


class SaintSemiSupLightningModule(pl.LightningModule):
    def __init__(self, transformer, embedding, optim, lr, 
                 weight_decay, prob_cutmix, alpha, 
                 lambda_pt, embed_dim, proj_head_dim, 
                 no_num, no_cat, cats, temperature,
                 task):
        super().__init__()
        self.transformer = transformer
        self.embedding = embedding
        self.prob_cutmix = prob_cutmix
        self.alpha = alpha
        self.lambda_pt = lambda_pt
        self.cats = cats
        self.cutmix = CutMix(self.prob_cutmix)
        self.mixup  = Mixup(self.alpha)
        
        self.lr = lr
        self.optim = optim                      # 'adamw'
        self.weight_decay = weight_decay
        self.task = task                        # {'ssl'}
        self.setup_criterion(embed_dim, proj_head_dim, no_num, no_cat, cats, temperature) 

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x
    
    def _shared_step(self, batch):
        xi, _    = batch                            # xi BS x (n+1)
        
        pi       =  self.embedding(xi)              # BS x (n+1) x d
        
        xi_prime =  self.cutmix(xi)                 # BS x (n+1)
        xi_prime_embed = self.embedding(xi_prime)   # BS x d x (n+1)
        pi_prime = self.mixup(xi_prime_embed)       # BS x (n+1) x d

        ri       = self.transformer(pi)             # BS x (n+1) x d
        ri_prime = self.transformer(pi_prime)       # BS x (n+1) x d
        constrastive_loss_step = self.constrastive_loss_fn(ri, ri_prime)
        denoising_loss_step = self.denoising_loss_fn(ri_prime, xi)

        loss = constrastive_loss_step + self.lambda_pt * denoising_loss_step

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        
        # log the outputs!
        self.log('train_loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        val_loss = self._shared_step(batch)
        
        # log the outputs!
        self.log(f'val_loss', val_loss, on_step=False, 
                 on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr,
                                        betas=(0.9, 0.999),
                                        weight_decay=self.weight_decay)
        else: # put as dummy optim
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': ReduceLROnPlateau(optimizer, 
                                                   patience=10, 
                                                   min_lr=1e-8, 
                                                   factor=0.5,
                                                   verbose=True),
                    'monitor': 'val_loss',
                }
            }

    def setup_criterion(self, embed_dim, proj_head_dim, no_num, no_cat, cats, temperature):
        self.constrastive_loss_fn = ConstrastiveLoss(embed_dim*(no_num+no_cat), 
                                                    proj_head_dim, 
                                                    temperature) # check the output dimension
        self.denoising_loss_fn = DenoisingLoss(no_num, no_cat, 
                                            cats, embed_dim)