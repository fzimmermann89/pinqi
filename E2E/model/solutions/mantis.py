import torch
from torch import nn, Tensor
from typing import Tuple, Optional
import numpy as np
from ...net import mantisUnet
from .base import Solution
from ...util import WarmupLR


class _mantis(torch.nn.Module):
    def __init__(self, n_data: int = 10, n_param: int = 2, xinitial_filters: int = 64, xfeature_growth=(2, 2, 2, 2, 1, 1), initial_values: Optional[Tensor] = None, adjust_x: bool = True):
        super().__init__()
        if adjust_x:
            n_param_net = n_param - 1
        else:
            n_param_net = n_param
        self.adjust_x = adjust_x
        self.xnet = mantisUnet(n_data, n_param_net, factors=xfeature_growth, filters=xinitial_filters)
        self.n_data = n_data
        if initial_values is not None:
            with torch.no_grad():
                if adjust_x:
                    self.xnet.dec[-1][-1].bias[:] = initial_values.ravel()[
                        (0, 2),
                    ]
                else:
                    self.xnet.dec[-1][-1].bias[:] = initial_values.ravel()

                self.xnet.dec[-1][-1].weight.normal_(0, 1e-6)

    def forward(self, k, enc):
        Ahk = enc.AH(k)
        y = Ahk
        x = self.xnet(y.abs())
        if self.adjust_x:
            x = torch.stack((x[:, 0], torch.zeros_like(x[:, 0]), x[:, 1]), dim=1)
        return ([x], [y], [], [])

    @property
    def device(self):
        return next(self.parameters()).device


class mantis(Solution):
    def __init__(
        self,
        Problem,
        xinitial_filters: int = 64,
        xfeature_growth=(2, 2, 2, 2, 1, 1),
        lr_net=1e-4,
        warmup_length: int = 1000,
        min_lr: float = 1e-5,
        lam_kloss: float = 1.0,
        outsidemaskstrength: float = 1.0,
        adjust_x: bool = True,
    ):
        super().__init__(Problem)
        self.net = _mantis(self.q.nOut, self.q.nIn, xinitial_filters, xfeature_growth, initial_values=self.q.initial_values, adjust_x=adjust_x)

        self.additional_info = dict(
            number_parameters=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.xnet.parameters())]),
            number_lambdas=0,
        )

    def forward(self, *x):
        return self.net(*x)

    def xloss(self, gtx, pred, mask):
        m = mask.float() + self.hparams.outsidemaskstrength * (~mask).float()
        loss = nn.functional.mse_loss(gtx * m, pred[0][-1] * m)
        return loss

    def kloss(self, k, pred, enc):
        t = pred[0][-1].clip(min=self.q.bounds[..., 0], max=self.q.bounds[..., 1])
        yp = self.q(t)
        kp = enc.A(yp)
        loss = nn.functional.mse_loss(torch.view_as_real(kp), torch.view_as_real(k))
        return loss

    def loss(self, gt, pred, mask):
        return self.xloss(gt, pred, mask)

    def adjust_x(self, x):
        xout = x.clone()
        xout[:, 0] = (xout[:, 0].square() + xout[:, 1].square()).sqrt()
        xout[:, 1].zero_()
        return xout

    def train_step(self, batch, opt, sched, batch_idx):
        opt.zero_grad()
        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)
        k = enc.A(y)
        k += enc.mask(noise)
        p = self(k, enc)
        if self.hparams.adjust_x:
            x = self.adjust_x(x)

        loss = self.hparams.lam_kloss * self.kloss(k, p, enc) + self.xloss(x, p, mask)
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
        # sched.step()
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            [
                {"params": self.net.xnet.parameters(), "lr": self.hparams.lr_net},
            ]
        )
        stepsTotal = self.trainer.fit_loop.max_epochs
        batches = self.trainer.estimated_stepping_batches
        # sched = WarmupLR(
        #     torch.optim.lr_scheduler.CosineAnnealingLR(optim, batches - self.hparams.warmup_length, eta_min=self.hparams.min_lr, verbose=False),
        #     self.hparams.min_lr,
        #     self.hparams.warmup_length,
        # )
        return [optim]  # [sched]
