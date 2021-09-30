import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import clones


class ContrastiveLoss(nn.Module):
    """Compute the contrastive loss ..."""

    def __init__(self, input_dim, proj_head_dim, temperature):
        super(ContrastiveLoss, self).__init__()
        self.projection_head_1 = nn.Sequential(
            nn.Linear(input_dim, proj_head_dim),
            nn.ReLU())

        self.projection_head_2 = nn.Sequential(
            nn.Linear(input_dim, proj_head_dim),
            nn.ReLU())

        self.temperature = temperature  # 1

    def contrastive_loss(self, zi, zi_prime):
        # zi dim = BS x proj_head_dim = zi_prime dim
        eps = 1e-7
        zi_product = torch.mm(zi, torch.t(zi_prime))  # BS x BS
        zi_product = zi_product / self.temperature

        exp_zi_prod = torch.exp(zi_product)  # BS x BS
        exp_zi_prod_sum = torch.sum(exp_zi_prod, dim=-1,
                                    keepdim=True)  # BS x 1

        return -1.0 * torch.sum(torch.log(
            F.relu(torch.diag(
                exp_zi_prod / exp_zi_prod_sum
            )) + eps))  # scalar

    def forward(self, ri, ri_prime):
        # ri dim = # BS x (n+1) x d = ri_prime dim
        ri = ri.reshape(ri.shape[0], -1)  # BS x (n+1)d
        ri_prime = ri_prime.reshape(ri_prime.shape[0], -1)  # BS x (n+1)d
        zi = self.projection_head_1(ri)  # BS x proj_head_dim
        zi_prime = self.projection_head_2(ri_prime)  # BS x proj_head_dim

        return self.contrastive_loss(zi, zi_prime)


class DenoisingLoss(nn.Module):
    def __init__(self, no_num, no_cat, cats, input_dim):
        super(DenoisingLoss, self).__init__()
        self.no_num = no_num
        self.no_cat = no_cat
        self.cats = cats
        self.cat_mlps = nn.ModuleList()
        for i in range(1, self.no_cat):
            self.cat_mlps.append(
                nn.Linear(input_dim, self.cats[i])
            )
        num_mlp = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU()
        )
        # one mlp per numerical feature
        self.num_mlps = clones(num_mlp, no_num)

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, ri_prime, xi):
        # xi (BS x n+1)
        # ri_prime (BS x n+1 x d)
        denoising_loss = 0.0
        num_loss = 0.0
        cat_loss = 0.0

        # pass each cat column through its mlp for a batch
        for feat_idx in range(1, self.no_cat):  # exclude [cls]
            # get the mlp for the feature
            ri_feat = self.cat_mlps[feat_idx - 1](
                ri_prime[:, feat_idx, :].squeeze()
            )  # BS x 1

            xi_feat = xi[:, feat_idx]  # BS x 1

            cat_loss += self.ce(ri_feat.float(),
                                xi_feat.long())

        for feat_idx in range(self.no_num):
            idx = self.no_cat + feat_idx

            # get the mlp for the feature
            ri_feat = self.num_mlps[feat_idx](ri_prime[:, idx, :])
            # BS x 1

            xi_feat = xi[:, idx]  # BS x 1

            num_loss += self.mse(ri_feat.squeeze().float(),
                                 xi_feat.float())

        denoising_loss = num_loss + cat_loss

        return denoising_loss
