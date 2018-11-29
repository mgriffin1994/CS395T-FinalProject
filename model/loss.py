# -*- coding: utf-8 -*-
"""loss.py

Loss functions

"""
import torch
import torch.nn.functional as F

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
