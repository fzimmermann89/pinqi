import torch
import pytorch_lightning as pl
from torch import nn
from typing import Tuple, Optional
import numpy as np

from .cg import CGO
from .bfgs import BFGS
from ...net.layers import ScaledTanh
from ...net import Unet
from .base import Solution
from ...util import WarmupLR
from ...util.bind import bind, invbind


class _SimpleSolve(torch.nn.Module):
    def __init__(self, q, threshold_channel_value: Tuple = (None, 0)):
        super().__init__()
        self.softplus = nn.Softplus(beta=10.0)
        self.q = q
        self.threshold_channel_value = threshold_channel_value

    def get_lambda(self, iteration, subproblem):
        return 0.0

    def forward(self, k, enc):
        Ahk = enc.AH(k)
        y = [self.solve_problem1(enc, Ahk, y0=Ahk)]
        x = [self.solve_problem2(self.q, y[-1], self.threshold_channel_value)]
        yreg = []
        xreg = []
        return (x, y, xreg, yreg)

    @staticmethod
    def solve_problem1(enc, Ahk, y0):
        r = CGO.apply(enc.AHA, Ahk, y0, y0, torch.zeros(1, device=Ahk.device), (-1, -2, -3))
        return r

    @staticmethod
    def solve_problem2(q, y, threshold_channel_value=None):
        x0 = q.initial_values.expand(y.shape[0], -1, *y.shape[-2:]).clone()
        r = BFGS.apply(lambda x: torch.abs(q(x)), y.abs(), x0, torch.zeros(1, device=x0.device), x0, q.bounds)
        if threshold_channel_value is not None and threshold_channel_value[0] is not None:
            mask = (r[:, threshold_channel_value[0]][:, None, ...].abs() < threshold_channel_value[1]).expand(r.shape)
            r[mask] = 0.0
        return r


class SimpleSolve(Solution):
    def __init__(self, Problem, threshold_channel_value: Tuple = (None, 0)):
        super().__init__(Problem)
        self.net = _SimpleSolve(self.q, threshold_channel_value)
        self.additional_info = dict(number_parameters=0, number_lambdas=0,)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, *x):
        return self.net(*x)

    def loss(self, gt, pred):
        loss = nn.functional.mse_loss(gt, pred[0][-1])
        return loss

    def training_step(self, batch, batch_idx):
        return torch.zeros(1)

    def configure_optimizers(self):
        return [], []
