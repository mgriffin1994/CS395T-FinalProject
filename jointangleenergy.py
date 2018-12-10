# -*- coding: utf-8 -*-
"""jointangleenergy.py

Get the energy of a joint angle using EBGAN

"""
import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os

import model.loss as module_loss
import model.metric as module_metric
import model.ebgan as EBGAN
from data_loader.data_loaders import JointAngleDataset
from utils import Logger

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class JointAngleEnergy(object):
    """Class to calculate the joint angle energy
    """
    def __init__(self, checkpoint=None):
        if not checkpoint:
            raise ValueError('No checkpoint provided!')
        self.checkpoint_path = checkpoint
        self.checkpoint = torch.load(checkpoint)
        self.config = self.checkpoint['config']
        self.generator = get_instance(EBGAN, 'g_arch', self.config)
        self.discriminator = get_instance(EBGAN, 'd_arch', self.config)
        self.generator.load_state_dict(self.checkpoint['0-statedict'])
        self.discriminator.load_state_dict(self.checkpoint['1-statedict'])

    def get_joint_angle_energy(self, joint_angles):
        """Get joint angle energies

        Use discriminator's autoencoder to calculate the energy.
        
        Parameters
        ---------
        joint_angles : list
            Joint angles of shape (N, 20)
        
        Returns
        -------
        np.array of Joint angle energies
            Shape (N, 1)

        """
        with torch.no_grad():
            # Note: divide joint angles by 2 because tanh activation used by 
            # autoencoder. Joint angles naturally between -1 and 1.5
            joint_angles = torch.FloatTensor(np.array(joint_angles) / 2)
            D = self.discriminator(joint_angles)[0]
            mse = np.array(F.mse_loss(joint_angles, D, reduction='none').data)
            return mse.sum(axis=1)

    def generate_samples(self, n):
        """Generate a sample with the generator

        Parameters
        ----------
        n : int
            The number of samples to generate

        Returns
        -------
        torch tensor of shape (n, 20)

        """
        z = torch.randn(n, self.generator.noise_dim)
        return np.array(self.generator(z).data) * 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-j', '--jointangles', default=None, type=str,
                           help='path to .csv file of joint angles (default: None)')
    parser.add_argument('-d', '--destination', default=None, type=str,
                           help='path to where to save the .json outputs')
    parser.add_argument('-g', '--generate', default=0, type=int,
                           help='the number of generated joint angle energies to include')
    args = parser.parse_args()

    jae = JointAngleEnergy(args.r)

    if args.jointangles:
        joint_angles = np.load(args.jointangles)
    else:
        print ("Joint angles file not provided. Instead, using database.")
        dataset = JointAngleDataset(1)
        joint_angles = np.array(dataset.data)

    joint_angle_energies = jae.get_joint_angle_energy(joint_angles)

    if args.generate > 0:
        generated_samples = jae.generate_samples(args.generate)
        generated_angle_energies = jae.get_joint_angle_energy(generated_samples)

    if args.destination:
        fpath = os.path.join(args.destination, 'joint_angle_energies.npy')
        np.save(fpath, joint_angle_energies)
        print ("Saved joint angle energies to {}".format(fpath))
        if args.generate > 0:
            fpath = os.path.join(args.destination, 'generator_angle_energies.npy')
            np.save(fpath, generated_angle_energies)
            print ("Saved generator angle energies to {}".format(fpath))
