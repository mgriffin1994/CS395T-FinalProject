# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.ebgan as EBGAN
from trainer import EBGANTrainer
from utils import Logger


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    print (len(data_loader))
    print (len(valid_data_loader))

    # build model architecture
    generator = get_instance(EBGAN, 'g_arch', config)
    generator.summary()

    discriminator = get_instance(EBGAN, 'd_arch', config)
    discriminator.summary()

    # get function handles of loss and metrics

    #loss = getattr(module_loss, config['loss'])
    #metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    g_trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    d_trainable_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    g_optimizer = get_instance(torch.optim, 'g_optimizer', config, g_trainable_params)
    d_optimizer = get_instance(torch.optim, 'd_optimizer', config, d_trainable_params)

    g_lr_scheduler = get_instance(torch.optim.lr_scheduler, 'g_lr_scheduler', config, g_optimizer)
    d_lr_scheduler = get_instance(torch.optim.lr_scheduler, 'd_lr_scheduler', config, d_optimizer)

    trainer = EBGANTrainer(generator, discriminator, [None],
                           g_optimizer, d_optimizer,
                           resume=resume,
                           config=config,
                           data_loader=data_loader,
                           valid_data_loader=valid_data_loader,
                           g_lr_scheduler=g_lr_scheduler,
                           d_lr_scheduler=d_lr_scheduler,
                           train_logger=train_logger)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)
