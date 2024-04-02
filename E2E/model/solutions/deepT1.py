import torch
import pytorch_lightning as pl
from torch import nn
from typing import Tuple, Optional, Union
import numpy as np

from .cg import CGO
from ...net import Unet, CRNN
from .base import Solution
from ...util import WarmupLR, softplus_inv


class _deepT1(torch.nn.Module):
    def __init__(
        self, q, initial_lambdas: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0), n_data: int = 10, n_param: int = 2, yfilters: int = 64, ylayer: int = 5, xlayer: int = 4, xinitial_filters: int = 64, phase: bool = True, fake_parameter=True
    ):
        super().__init__()
        self.softplus = nn.Softplus(beta=5.0)

        self.ynet = CRNN(filters=yfilters, layers=ylayer)
        self.lambdas = nn.Parameter(softplus_inv(torch.as_tensor(initial_lambdas, dtype=torch.float32), beta=self.softplus.beta).requires_grad_(True))
        self.q = q
        self.n_data = n_data
        self.phase = phase
        if fake_parameter:
            n_param -= 1
        self.xnet = Unet(
            2,
            channels_in=n_data * (2 if phase else 1),
            channels_out=n_param,
            filters=xinitial_filters,
            change_filters_last=False,
            layer=xlayer,
            feature_growth=lambda x: 1 if (x == 0 or x > xlayer - 1) else 2,
            up_mode="conv_reduce",
        )
        self.fake_parameter = fake_parameter

    @staticmethod
    def DC(enc, Ahk, x, l):
        r = CGO.apply(enc.AHA, Ahk, x, Ahk, l, (-1, -2, -3))
        return r

    def forward(self, k, enc, e2e=False, onlyy=False):

        Ahk = enc.AH(k)
        x, xreg, yreg = [], [], []
        DC = [lambda x: self.DC(enc, Ahk, x, self.softplus(l)) for l in self.lambdas]
        y = self.ynet(Ahk, DC)

        if onlyy:
            return (x, y, xreg, yreg)

        finaly = y[-1]
        if not e2e:
            finaly = finaly.detach()
        x = self.xnet(self.reshape_c2r_y(finaly))
        if self.fake_parameter:
            x = torch.stack((x[:, 0], torch.zeros_like(x[:, 0]), x[:, 1]), dim=1)
        return ([x], y, xreg, yreg)

    def reshape_c2r_y(self, y):
        if self.phase:
            return torch.view_as_real(y).moveaxis(-1, 1).reshape(y.shape[0], -1, *y.shape[-2:])
        else:
            return y.abs()

    @property
    def device(self):
        return next(self.parameters()).device


class deepT1(Solution):
    def __init__(
        self,
        Problem,
        initial_lambdas: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
        yfilters: int = 64,
        ylayer: int = 5,
        xlayer: int = 4,
        xinitial_filters: int = 64,
        xfeature_growth=[2, 2, 2, 1],
        outsidemaskstrength: float = 1e-4,
        lr_net: float = 1e-4,
        lr_ynet: float = 1e-4,
        lr_lam: float = 1e-4,
        pretrain_length: Union[int, float] = None,
        fake_parameter: bool = True,
    ):
        super().__init__(Problem)
        self.net = _deepT1(self.q, initial_lambdas, self.q.nOut, self.q.nIn, yfilters, ylayer, xlayer, xinitial_filters, phase=False, fake_parameter=fake_parameter)
        self.additional_info = dict(
            number_parameters=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.xnet.parameters())]) + sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.ynet.parameters())]),
            number_lambdas=len(self.net.lambdas),
        )

    def forward(self, *x, **kwargs):
        return self.net(*x, **kwargs)

    def xloss(self, gtx, gty, pred, mask):
        mask = mask + self.hparams.outsidemaskstrength
        p = pred[0][-1]
        if self.hparams.fake_parameter:
            mask = mask[:, 0]
            gt_m0 = (gtx[:, 0].square() + gtx[:, 1].square()).sqrt()
            loss = nn.functional.l1_loss(mask * p[:, 0], mask * gt_m0) + nn.functional.l1_loss(mask * p[:, 2], mask * gtx[:, 2])
        else:
            loss = nn.functional.l1_loss(mask * p, mask * gtx)
        return loss

    def yloss(self, gtx, gty, pred, mask):
        loss = nn.functional.l1_loss(gty.abs(), pred[1][-1].abs())
        return loss

    def loss(self, gt, pred, mask):
        return self.xloss(gt, None, pred, mask)

    def step(self, batch, opt, sched, pre=False):
        opt.zero_grad()

        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)
        k = enc.A(y)
        k += enc.mask(noise)

        if pre:
            p = self(k, enc, e2e=False, onlyy=True)
            loss = self.yloss(x, y, p, mask)
        else:
            p = self(k, enc, e2e=False, onlyy=False)
            loss = self.xloss(x, y, p, mask)
        self.manual_backward(loss)

        try:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0, error_if_nonfinite=True)
        except Exception as e:
            print("grad error:", e)
            for n, p in self.net.named_parameters():
                if not torch.all(torch.isfinite(p.grad)):
                    print(n, p.grad.ravel()[:5])
                    p.grad.fill_(0.0)
        opt.step()
        if sched is not None:
            sched.step()
        return loss

    def pretrain_step(self, batch, opt, sched, batch_idx):
        return self.step(batch, opt, sched, True)

    def train_step(self, batch, opt, sched, batch_idx):
        return self.step(batch, opt, sched, False)

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            [
                {"params": self.net.xnet.parameters(), "lr": self.hparams.lr_net},
            ]
        )
        optimPre = torch.optim.Adam(
            [
                {"params": self.net.ynet.parameters(), "lr": self.hparams.lr_ynet},
                {
                    "params": [self.net.lambdas],
                    "lr": self.hparams.lr_lam,
                },
            ],
        )
        stepsTotal = self.trainer.fit_loop.max_epochs
        if self.hparams.pretrain_length < 1:
            self.hparams.pretrain_length = int(self.hparams.pretrain_length * stepsTotal)
        self.trainer.fit_loop.max_epochs = stepsTotal - self.hparams.pretrain_length
        batches = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = self.hparams.pretrain_length
        batchesPre = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = stepsTotal
        # sched = WarmupLR(
        #     torch.optim.lr_scheduler.CosineAnnealingLR(optim, batches - self.hparams.warmup_length, eta_min=self.hparams.min_lr, verbose=False),
        #     self.hparams.min_lr,
        #     self.hparams.warmup_length,
        # )

        # warmupPre = max(int(self.hparams.warmup_length / batches * batchesPre), 100)
        # schedPre = WarmupLR(
        #     torch.optim.lr_scheduler.CosineAnnealingLR(optimPre, batchesPre - warmupPre, eta_min=self.hparams.min_lr, verbose=False),
        #     self.hparams.min_lr,
        #     warmupPre,
        # )
        # sched,schedPre = None,None
        return [optim, optimPre]  # , [sched, schedPre]
