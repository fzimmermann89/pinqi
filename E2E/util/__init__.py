from .warmup import WarmupLR
from .mask import cutnan, AddFillMask, RemoveFillMask
from .rnd import trunc_norm, random_poly2d, random_gaussians, gamma_normal_noise
from .gpuman import use_emptiest_gpu
from .tensorfunctions import interleave, softplus_inv
from .cacheDS import cacheDS
from .filterDS import filterDS
from .tensorgrad import GradOperators, SmoothTV
from .cp import chambolle_pock
from .cgtv import conjugate_gradient_TV
from .shepp_logan import shepp_logan

def seed_numpy(id: int):
    """
    Seed numpy in workers
    """
    import numpy as np
    import torch

    np.random.seed((id + torch.initial_seed()) % np.iinfo(np.int32).max)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)