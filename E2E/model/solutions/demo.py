import torch
import pytorch_lightning as pl
from torch import nn
from typing import Tuple, Optional
import numpy as np
from ...net import Unet
from .base import Solution
from ...util import WarmupLR
from ...util.bind import bind, invbind


class DNet(nn.Module):
    def __init__(self, mem=0, channels_in=16, n_other=1, filters=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d((1 + mem + 2 + n_other) * channels_in, filters, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
            nn.ReLU(),
            nn.Conv3d(filters, filters, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
            nn.ReLU(),
            nn.Conv3d(filters, (1 + mem) * channels_in, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
        )

    def forward(self, h, Ay, k, *other):
        x = torch.cat((h, Ay, k, *other), 1)
        h = h + self.block(x)
        return h


class PNet(nn.Module):
    def __init__(self, mem=0, channels_in=2, n_other=1, filters=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d((mem + n_other + 2) * channels_in, filters, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
            nn.ReLU(),
            nn.Conv3d(filters, filters, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
            nn.ReLU(),
            nn.Conv3d(filters, (1 + mem) * channels_in, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
        )

    def forward(self, y, AHh, *other):
        x = torch.cat((AHh, y, *other), 1)
        y = y + self.block(x)
        return y


class _demo(nn.Module):
    def __init__(
        self,
        q,
        n_param: int,
        n_data: int,
        n_coil: int,
        iterations: int = 5,
        pd_iterations: int = 1,
        mem_primal: int = 0,
        mem_dual: int = 0,
        p_filters: int = 32,
        d_filters: int = 32,
        xinital_filters: int = 16,
        xlayers: int = 2,
    ):

        super().__init__()
        self.mem_primal = mem_primal
        self.mem_dual = mem_dual
        self.primal_nets = nn.ModuleList([nn.ModuleList([PNet(mem_primal, filters=p_filters) for i in range(pd_iterations)]) for j in range(iterations)])
        self.dual_nets = nn.ModuleList([nn.ModuleList([DNet(mem_dual, n_coil * 2, filters=d_filters) for i in range(pd_iterations)]) for j in range(iterations)])
        self.map_net = Unet(2, n_data * 2, n_param, xlayers, filters=xinital_filters, change_filters_last=False, feature_growth=lambda i: 1 if i == 0 else 2, skip_add=True)
        self.q = q

    @staticmethod
    def c2r(c):
        r = torch.view_as_real(c).moveaxis(-1, 1)
        return r

    @staticmethod
    def r2c(r):
        c = torch.view_as_complex(r.moveaxis(1, -1).contiguous())
        return c

    def forward(self, k, enc):
        AHk = enc.AH(k)
        kr = self.c2r(k).swapaxes(2, 3)
        kr = kr.flatten(1, 2)
        h = [torch.zeros(kr.shape[0], (self.mem_dual + 1) * kr.shape[1], *kr.shape[2:], device=AHk.device)]
        y = [torch.cat([self.c2r(AHk)] * (self.mem_primal + 1), 1)]
        x = []
        qx = []
        for j, PDNets in enumerate(zip(self.dual_nets, self.primal_nets)):
            map_input = y[-1][:, :2]
            map_input = map_input.reshape(map_input.shape[0], -1, *map_input.shape[-2:])
            x.append(self.map_net(map_input))
            qx.append(self.q(x[-1]))
            qxr = self.c2r(qx[-1])
            Aqx = self.c2r(enc.A(qx[-1])).swapaxes(2, 3).flatten(1, 2)
            for DualNet, PrimalNet in zip(*PDNets):
                Ay = self.c2r(enc.A(self.r2c(y[-1][:, :2]))).swapaxes(2, 3).flatten(1, 2)
                h.append(DualNet(h[-1], Ay, kr, Aqx))
                newk = h[-1][:, : k.shape[2] * 2]
                newk = self.r2c(newk.reshape(newk.shape[0], 2, -1, *newk.shape[2:]).swapaxes(2, 3))
                AHnk = self.c2r(enc.AH(newk))

                y.append(PrimalNet(y[-1], AHnk, qxr))
        map_input = y[-1][:, :2]
        map_input = map_input.reshape(map_input.shape[0], -1, *map_input.shape[-2:])
        x.append(self.map_net(map_input))
        return x, [i[:, :2] for i in y], [], qx


class demo(Solution):
    def __init__(
        self,
        Problem,
        n_coil: int,
        iterations: int = 5,
        pd_iterations: int = 1,
        mem_primal: int = 0,
        mem_dual: int = 0,
        p_filters: int = 32,
        d_filters: int = 32,
        xinital_filters: int = 16,
        xlayers: int = 2,
        lr_net=1e-3,
        lr_lam=1e-2,
        warmup_length: int = 1000,
        pretrain_length: int = 1,
        min_lr: float = 1e-5,
    ):
        """
        https://i-mri.org/pdf/10.13104/imri.2021.25.4.300
        """

        super().__init__(Problem)
        self.net = _demo(
            self.q,
            self.q.nIn,
            self.q.nOut,
            n_coil=n_coil,
            iterations=iterations,
            pd_iterations=pd_iterations,
            mem_primal=mem_primal,
            mem_dual=mem_dual,
            p_filters=p_filters,
            d_filters=d_filters,
            xinital_filters=xinital_filters,
            xlayers=xlayers,
        )

        self.additional_info = dict(number_parameters=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.parameters())]), number_lambdas=0)

    def forward(self, *x):
        return self.net(*x)

    def xloss(self, gtx, gty, pred, mask):
        mask = mask + 1e-4
        loss = nn.functional.mse_loss(mask * gtx, mask * pred[0][-1])
        return loss

    def yloss(self, gtx, gty, pred, mask):
        mask = mask + 1e-2
        loss = nn.functional.mse_loss(mask * gty.abs(), mask * pred[1][-1].abs())
        return loss

    def loss(self, gt, pred, mask):
        return self.xloss(gt, None, pred, mask)

    def step(self, batch, opt, sched, pre=False):
        if pre:
            lossfunction = lambda *args: self.yloss(*args) + self.xloss(*args)
        elif self.hparams.e2e:
            lossfunction = self.xloss
        else:
            lossfunction = lambda *x: self.xloss(*x) + self.yloss(*x)
        opt.zero_grad()
        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)
        k = enc.A(y)
        k += enc.mask(noise)
        p = self(k, enc)
        loss = lossfunction(x, y, p, mask)
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
                {"params": self.net.ynet.parameters(), "lr": self.hparams.lr_net},
                {"params": self.net.lambdas.parameters(), "lr": self.hparams.lr_lam},
            ]
        )
        optimPre = torch.optim.Adam(
            [
                {"params": self.net.ynet.parameters(), "lr": self.hparams.lr_net},
                {"params": self.net.lambdas.parameters(), "lr": self.hparams.lr_lam},
            ]
        )
        # print("trainbatch", self.trainer.num_training_batches)
        # print("before", self.trainer.estimated_stepping_batches)
        stepsTotal = self.trainer.fit_loop.max_epochs
        self.trainer.fit_loop.max_epochs = stepsTotal - self.hparams.pretrain_length
        batches = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = self.hparams.pretrain_length
        batchesPre = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = stepsTotal
        # print("batches", batches, "batchesPre", batchesPre, "total", stepsTotal)
        sched = WarmupLR(
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, batches - self.hparams.warmup_length, eta_min=self.hparams.min_lr, verbose=False),
            self.hparams.min_lr,
            self.hparams.warmup_length,
        )

        warmupPre = max(int(self.hparams.warmup_length / batches * batchesPre), 100)
        schedPre = WarmupLR(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimPre, batchesPre - warmupPre, eta_min=self.hparams.min_lr, verbose=False),
            self.hparams.min_lr,
            warmupPre,
        )
        return [optim, optimPre], [sched, schedPre]
