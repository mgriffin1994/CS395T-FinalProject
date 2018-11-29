# -*- coding: utf-8 -*-
"""ebgan.py

Trainer for ebgan

"""
from base import BaseTrainer

class EBGAN(BaseTrainer):
    def __init__(self, generator, discriminator, loss, metrics,):
        super(EBGAN, self).__init__(generator)

