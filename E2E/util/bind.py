import torch
from torch import Tensor
from typing import Optional


def invsp(y, beta: float = 1.0):
    return y + (1 - (y * beta).neg().exp()).log() / beta


#@torch.jit.script
def bind(x, bounds: Optional[Tensor] = None):
    if bounds is None:
        return x.clone()
    lower, upper = bounds[..., 0], bounds[..., 1]
    haslower = torch.isfinite(lower)
    hasupper = torch.isfinite(upper)
    lower, upper = torch.nan_to_num(lower, posinf=1e30, neginf=-1e30, nan=-1e30), torch.nan_to_num(upper, posinf=1e30, neginf=-1e30, nan=1e30)
    r = x.clone()
    r = torch.where(hasupper & ~haslower, upper - torch.nn.functional.softplus(upper - x, beta=8), r)
    r = torch.where(haslower & ~hasupper, torch.nn.functional.softplus(x - lower, beta=8) + lower, r)
    r = torch.where(haslower & hasupper, torch.tanh(x / ((upper - lower) / 2)) * ((upper - lower) / 2) + (lower + upper) / 2, r)
    return r


#@torch.jit.script
def invbind(x, bounds: Optional[Tensor] = None):
    if bounds is None:
        return x.clone()
    lower, upper = bounds[..., 0], bounds[..., 1]
    haslower = torch.isfinite(lower)
    hasupper = torch.isfinite(upper)
    lower, upper = torch.nan_to_num(lower, posinf=1e30, neginf=-1e30, nan=-1e30), torch.nan_to_num(upper, posinf=1e30, neginf=-1e30, nan=1e30)
    x = x.clone().clamp(min=lower + 1e-6, max=upper - 1e-6)
    r = x
    r = torch.where(haslower & hasupper, (upper - lower) / 2 * torch.arctanh((lower + upper - 2 * x) / (lower - upper)), r)
    r = torch.where(hasupper & ~haslower, upper - invsp(upper - x, 8.0), r)
    r = torch.where(haslower & ~hasupper, invsp(x - lower, 8.0) + lower, r)
    return r
