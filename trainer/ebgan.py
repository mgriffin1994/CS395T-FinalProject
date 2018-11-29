# -*- coding: utf-8 -*-
"""ebgan.py

Trainer for ebgan

"""
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F

from base import BaseTrainer

def repelling_regularizer(s1, s2):
    """Pulling away term
    
    Pulling away term to avoid mode collapse

    Inputs
    ------
    s1 : Torch tensor
    s2 : Torch tensor

    Returns
    -------
    Torch tensor
        Repelling regularizer loss

    """
    n = s1.size(0)
    s1 = F.normalize(s1, p=2, dim=1)
    s2 = F.normalize(s2, p=2, dim=1)

    S1 = s1.unsqueeze(1).repeat(1, s2.size(0), 1)
    S2 = s2.unsqueeze(0).repeat(s1.size(0), 1, 1)

    f_PT = S1.mul(S2).sum(-1).pow(2)
    f_PT = torch.tril(f_PT, -1).sum().mul(2).div((n*(n-1)))

    #f_PT = (S1.mul(S2).sum(-1).pow(2).sum(-1)-1).sum(-1).div(n*(n-1))
    return f_PT

class EBGANTrainer(BaseTrainer):
    """EBGAN Trainer Class"""
    def __init__(self, generator, discriminator, metrics, 
                 g_optimizer, d_optimizer, resume, config, data_loader,
                 pt_regularization=0.1,valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(EBGAN, self).__init__([generator, discriminator], metrics, optimizer, resume, config, train_logger)
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))


    def _discriminator_loss(self, inputs, target):
        """Discriminator loss"""
        # TODO - consider having this as a changeable parameter
        return F.mse_loss(inputs, target)
    
    def _generator_loss(self, x_fake, D_fake, D_latent):
        """Generator loss"""
        G_loss_PT = repelling_regularizer(D_latent, D_latent)
        G_loss_fake = F.mse_loss(x_fake, D_fake)
        return G_loss_PT + G_loss_fake

    def _sample_z(self, batch_size, dim, dist='normal'):
        """Sample noise"""
        if dist == 'normal':
            return torch.randn(batch_size, dim)
        elif dist == 'uniform':
            return torch.rand(batch_size, dim).mul(2).add(-1)
        else:
            return None

    def _eval_metrics(self, output, target):
        pass

    def _train_epoch(self, epoch):
        """Training logic for an epoch

        Inputs
        ------
        epoch : int
            The current training epoch
        
        Returns
        -------
        Log with information to save

        """

        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            # Train the discriminator
            x_real = inputs.to(self.device)
            D_real = self.discriminator(x_real)[0]
            D_loss_real = self._discriminator_loss(x_real, D_real)

            z = self.sample_z(self.data_loader.batch_size, self.generator.noise_dim)
            z = z.to(self.device)
            x_fake = self.generator(z)

            D_fake, D_latent = self.discriminator(x_fake)

            G_loss = self._generator_loss(x_fake, D_fake, D_latent)

            
           


