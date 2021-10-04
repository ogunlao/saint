import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl

from utils.augment import CutMix, Mixup
from utils.loss import ContrastiveLoss, DenoisingLoss
from utils.utils import Metric

class SaintSupLightningModule(pl.LightningModule):
    def __init__(self, transformer, embedding, fc, optim, lr,
                 weight_decay, task, num_output, cls_token_idx, 
                 freeze_encoder=False, metric='auroc', **kwargs
                 ):
        super().__init__()
        self.transformer = transformer
        self.embedding = embedding
        if freeze_encoder:
            # use model as feature extractor
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
            self.embedding.eval()
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.fc = fc
        self.lr = lr
        self.optim = optim                                      # 'adamw'
        self.weight_decay = weight_decay
        self.num_classes = None if num_output == 1 else num_output
        self.task = task                                        
        self.cls_token_idx = cls_token_idx
        self.metric=metric
        self.setup_criterion()

        self.train_metric = Metric(self.num_classes).get_metric(self.metric)
        self.valid_metric = Metric(self.num_classes).get_metric(self.metric)
        self.test_metric = Metric(self.num_classes).get_metric(self.metric)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
  
        out = self.fc(x[:, self.cls_token_idx, :].squeeze())       # BS x embed_dim
        return out
    
    def _shared_step(self, batch, metric_fn):
        x, targets = batch
        targets = targets.squeeze()
        
        x = self.embedding(x)
        x = self.transformer(x)

        outputs = self.fc(x[:, self.cls_token_idx, :]).squeeze()    # BS x embed_dim
        
        # Need to cast to cater for either binary or multi-class loss
        targets = targets.float() if self.num_classes is None else targets
        
        loss = self.criterion(outputs, targets)
        
        with torch.no_grad():
            if self.num_classes is None:
                preds = torch.sigmoid(outputs)
            else:
                preds = nn.functional.softmax(outputs, dim=1)

        metric_fn.update(preds, targets.long())
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.train_metric)
        
        # log the outputs!
        self.log('train_loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        self.log(f'train_{self.metric}_epoch', self.train_metric.compute(), prog_bar=True,)

        # reset after each epoch
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        val_loss = self._shared_step(batch, self.valid_metric)
        
        # log the outputs!
        self.log(f'val_loss', val_loss, on_step=False, 
                 on_epoch=True, prog_bar=True, logger=True)
        
    def validation_epoch_end(self, validation_step_outputs):
        self.log(f'val_{self.metric}_epoch', self.valid_metric.compute(), prog_bar=True,)

        # reset after each epoch
        self.valid_metric.reset()

    def test_step(self, batch, batch_idx):
        test_loss = self._shared_step(batch, self.test_metric)
        
        # log the outputs!
        self.log('test_loss', test_loss, on_step=False, 
                 on_epoch=True, prog_bar=True, logger=True)
    
    def test_epoch_end(self, test_outs):
        self.log(f'test_{self.metric}_best_epoch', self.test_metric.compute())

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
                 task, **kwargs):
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
    
    def _shared_step(self, batch, step):
        xi, _    = batch                            # xi BS x (n+1)
        pi       =  self.embedding(xi)              # BS x (n+1) x d
        
        xi_prime = self.cutmix(xi)                 # BS x (n+1)
        xi_prime_embed = self.embedding(xi_prime)   # BS x d x (n+1)
        pi_prime = self.mixup(xi_prime_embed)       # BS x (n+1) x d

        ri       = self.transformer(pi)             # BS x (n+1) x d
        ri_prime = self.transformer(pi_prime)       # BS x (n+1) x d

        contrastive_loss_step = self.contrastive_loss_fn(ri, ri_prime)
        denoising_loss_step = self.denoising_loss_fn(ri_prime, xi)
        loss = contrastive_loss_step + self.lambda_pt * denoising_loss_step

        self.log(f'{step}_contras_loss', contrastive_loss_step, on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{step}_denoise_loss', denoising_loss_step, on_step=True, 
                 on_epoch=True, prog_bar=False, logger=True)
        # log the outputs!
        self.log(f'{step}_loss', loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'train')
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        val_loss = self._shared_step(batch, 'val')

        return {'val_loss': val_loss}

    def configure_optimizers(self):
        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr,
                                        betas=(0.9, 0.999),
                                        weight_decay=self.weight_decay)
        else: # default optim
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
        self.contrastive_loss_fn = ContrastiveLoss(embed_dim*(no_num+no_cat),
                                                    proj_head_dim, 
                                                    temperature)
        self.denoising_loss_fn = DenoisingLoss(no_num, no_cat, 
                                            cats, embed_dim)