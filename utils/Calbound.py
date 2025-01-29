import math
import torch
from tools import *

def calc_lower_bound(z_1, z_2, pos, temperature = 0.2):
    EOS = 1e-10    
    z_1=z_1.cuda()
    z_2=z_2.cuda()
    pos=pos.cuda()

    matrix_1 = torch.exp(sim_con(z_1, z_2, temperature))
    matrix_2 = matrix_1.t()

    matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EOS)
    lori_1 = -torch.log(torch.clamp(matrix_1.mul(pos).sum(dim=-1), min=1e-10)).mean()

    matrix_2 = matrix_2 / (torch.sum(matrix_2, dim=1).view(-1, 1) + EOS)
    lori_2 = -torch.log(torch.clamp(matrix_2.mul(pos).sum(dim=-1), min=1e-10)).mean()

    return (lori_1 + lori_2) / 2