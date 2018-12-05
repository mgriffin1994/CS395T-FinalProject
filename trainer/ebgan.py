# -*- coding: utf-8 -*-
"""ebgan.py

Trainer for ebgan

"""
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
import time

from base import BaseTrainer
from utils.logger import AverageMeter

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
                 margin=20, pt_regularization=0.1, valid_data_loader=None, g_lr_scheduler=None, d_lr_scheduler=None, train_logger=None):
        super(EBGANTrainer, self).__init__([generator, discriminator], metrics, [g_optimizer, d_optimizer], resume, config, train_logger)
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.g_lr_scheduler = g_lr_scheduler
        self.d_lr_scheduler = d_lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.margin = margin


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
        raise NotImplementedError

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
        if self.verbosity > 2:
            print ("Train at epoch {}".format(epoch))

        self.generator.train()
        self.discriminator.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        dlr = AverageMeter()
        dlf = AverageMeter()
        g_loss = AverageMeter()

        end_time = time.time()
        for batch_idx, (data, labels) in enumerate(self.data_loader):
            data_time.update(time.time() - end_time)

            # Train the discriminator
            x_real = data.to(self.device)
            D_real = self.discriminator(x_real)[0]

            D_loss_real = self._discriminator_loss(x_real, D_real)

            z = self._sample_z(self.data_loader.batch_size, self.generator.noise_dim)
            z = z.to(self.device)
            x_fake = self.generator(z)
            D_fake = self.discriminator(x_fake.detach())[0]
            D_loss_fake = self._discriminator_loss(x_fake, D_fake)

            D_loss = D_loss_real
            if D_loss_fake.item() < self.margin:
                D_loss += (self.margin - D_loss_fake)

            self.d_optimizer.zero_grad()
            D_loss.backward()
            self.d_optimizer.step()

            # Train the generator
            z = self._sample_z(self.data_loader.batch_size, self.generator.noise_dim)
            z = z.to(self.device)
            x_fake = self.generator(z)
            D_fake, D_latent = self.discriminator(x_fake)

            G_loss = self._generator_loss(x_fake, D_fake, D_latent)

            self.g_optimizer.zero_grad()
            G_loss.backward()
            self.g_optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            dlr.update(D_loss_real.item(), x_real.size(0))
            dlf.update(D_loss_fake.item(), x_real.size(0))
            g_loss.update(G_loss.item(), z.size(0))

            if self.verbosity >= 2:
                info = ('Epoch: {} [{}/{} ({:.0f}%)]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Discriminator Loss (Real) {dlr.val:.4f} ({dlr.avg:.4f})\t'
                      'Discriminator Loss (Fake) {dlf.val:.4f} ({dlf.avg:.4f})\t'
                      'Generator Loss (Real) {g_loss.val:.4f} ({g_loss.avg:.4f})\t').format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    dlr=dlr,
                    dlf=dlf,
                    g_loss=g_loss)
                if batch_idx % self.log_step == 0:
                    self.logger.info(info)
                    self.writer.add_image('inp', make_grid(data.cpu(), nrow=8, normalize=True))
                print (info)

        log = {
            'dlr' : dlr.avg,
            'dlf' : dlf.avg,
            'g_loss' : g_loss.avg,
            #'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.g_lr_scheduler is not None:
            self.g_lr_scheduler.step()
        if self.d_lr_scheduler is not None:
            self.d_lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """Validation after training an epoch"""
        self.discriminator.eval()
        self.generator.eval()

        if self.verbosity > 2:
            print ("Validation at epoch {}".format(epoch))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        dlr = AverageMeter()
        dlf = AverageMeter()
        g_loss = AverageMeter()

        end_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.valid_data_loader):
                data_time.update(time.time() - end_time)

                x_real = data.to(self.device)
                D_real = self.discriminator(x_real)[0]
                D_loss_real = self._discriminator_loss(x_real, D_real)

                z = self._sample_z(self.data_loader.batch_size, self.generator.noise_dim)
                z = z.to(self.device)
                x_fake = self.generator(z)
                D_fake = self.discriminator(x_fake.detach())[0]
                D_loss_fake = self._discriminator_loss(x_fake, D_fake)

                D_loss = D_loss_real
                if D_loss_fake.item() < self.margin:
                    D_loss += (self.margin - D_loss_fake)

                z = self._sample_z(self.data_loader.batch_size, self.generator.noise_dim)
                z = z.to(self.device)
                x_fake = self.generator(z)
                D_fake, D_latent = self.discriminator(x_fake)

                G_loss = self._generator_loss(x_fake, D_fake, D_latent)

                batch_time.update(time.time() - end_time)

                dlr.update(D_loss_real.item(), x_real.size(0))
                dlf.update(D_loss_fake.item(), x_real.size(0))
                g_loss.update(G_loss.item(), z.size(0))

                if self.verbosity >= 2:
                    print ('Epoch: {} [{}/{} ({:.0f}%)]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Discriminator Loss (Real) {dlr.val:.4f} ({dlr.avg:.4f})\t'
                          'Discriminator Loss (Fake) {dlf.val:.4f} ({dlf.avg:.4f})\t'
                          'Generator Loss {g_loss.val:.4f} ({g_loss.avg:.4f})\t'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        dlr=dlr,
                        dlf=dlf,
                        g_loss=g_loss))

        log = {
            'dlr' : dlr.avg,
            'dlf' : dlf.avg,
            'g_loss' : g_loss.avg,
        }

        return log

    def sample_generator():
        # TODO
        pass
        
