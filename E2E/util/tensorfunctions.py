import torch


def softplus_inv(y, beta=1.0):
    return y + torch.log(-torch.expm1(-y * beta)) / beta


def interleave(tensors, dim):
    shape = tensors[0].shape
    return torch.stack(tensors, dim=dim + 1).view(*shape[:dim], -1, *shape[dim + 1 :])
