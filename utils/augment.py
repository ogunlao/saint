import torch


class CutMix:
    """Applies cutmix in the feature space to the features"""

    def __init__(self, prob_cutmix):
        super(CutMix, self).__init__()
        self.prob_cutmix = prob_cutmix

    def __call__(self, x_i):                                            # x_i # BS x (n+1)

        shuffled_index = torch.randperm(x_i.shape[0])                   # BS
        x_a = x_i[shuffled_index]                                       # BS x (n+1)

        prob_matrix = torch.ones(x_i.shape) * (1 - self.prob_cutmix)    # BS x (n+1)
        
        m_binary_matrix = torch.empty_like(x_i)                         # BS x (n+1)
        torch.bernoulli(prob_matrix, out=m_binary_matrix)               # BS x (n+1)
        
        xi_cutmix = m_binary_matrix * x_i + (1 - m_binary_matrix) * x_a # BS x (n+1)
        
        return xi_cutmix                                                # BS x (n+1)
    
class Mixup:
    "Applies mixup to feature embeddings"

    def __init__(self, alpha):
        super(Mixup, self).__init__()
        self.alpha = alpha

    def __call__(self, xi_embed):                                       # BS x (n+1)
        shuffled_index = torch.randperm(xi_embed.shape[0])              # BS
        xb_prime = xi_embed[shuffled_index]                             # BS x d x (n+1)
        p_i = self.alpha * xi_embed + (1 - self.alpha) * xb_prime       # BS x d x (n+1)

        return p_i