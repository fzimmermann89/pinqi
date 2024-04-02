import torch
import pytorch_lightning as pl
from torch import nn
from typing import Tuple, Optional, Union
import numpy as np
import functorch

from ...net.blocks import CBlock
from .base import Solution
from ...util import WarmupLR, softplus_inv
from .cg import cg


class DR(nn.Module):
    def __init__(self, parameters: Tuple[int, ...] = [2, 1], layer: int = 5, filters: int = 64, residual: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([CBlock((p, *(layer * (filters,)), p), norm="instance", bias="last", final_activation=False, final_norm=False) for p in parameters])
        self.splits = parameters
        self.residual = residual
        for b in self.blocks:
            torch.nn.init.zeros_(b[-1].bias)
            torch.nn.init.zeros_(b[-1].weight)

    def forward(self, x):
        s = torch.split(x, self.splits, dim=1)
        ret = torch.cat([b(p) for p, b in zip(s, self.blocks)], 1)
        if self.residual:
            ret = ret + x
        return ret


class DM(nn.Module):
    def __init__(self, n_data, parameters: Tuple[int, ...], layer: int = 5, filters: int = 64):
        super().__init__()
        self.block = CBlock((n_data * 2, *(layer * (filters,)), sum(parameters)), norm="instance", kernel_size=1, bias="last", final_activation=False, final_norm=False)

    def forward(self, x):
        x = torch.view_as_real(x).moveaxis(-1, 1).reshape(x.shape[0], -1, *x.shape[2:])
        x = self.block(x)
        return x


class _dopamine(torch.nn.Module):
    def __init__(
        self,
        q,
        initial_lambdas: Tuple[float, float] = (0.1, 0.1),
        iterations: int = 10,
        n_data: int = 10,
        parameters: Tuple[int, ...] = [2, 1],
        xlayer: int = 5,
        xfilters: int = 64,
        ylayer: int = 5,
        yfilters: int = 64,
        y0iterations: int = 0,
        tanh_bound: Union[bool, float] = True,
    ):
        super().__init__()
        self.ynet = DM(n_data, parameters, layer=ylayer, filters=yfilters)
        self.xnet = DR(parameters, layer=xlayer, filters=xfilters)
        self.q = q
        self.n_data = n_data
        self.softplus = nn.Softplus(beta=5.0)
        self.lambdas = nn.ParameterList([nn.Parameter(softplus_inv(torch.as_tensor(initial_lambdas, dtype=torch.float32), beta=self.softplus.beta)) for i in range(iterations)])
        self.y0iterations = y0iterations
        if tanh_bound:
            s = float(tanh_bound)
            self.reg_dp = lambda x: torch.tanh(x * (1 / s)) * s
        else:
            self.reg_dp = lambda x: x

    def forward(self, k, enc):
        Ahk = enc.AH(k)
        xreg, yreg = (
            [],
            [],
        )

        if self.y0iterations > 0:
            with torch.no_grad():
                y = [cg(enc.AHA, Ahk, Ahk, maxiter=self.y0iterations, dims=(-1, -2, -3))]
        else:
            y = [Ahk]
        x0 = self.ynet(y[-1])
        p = x0.requires_grad_(True)
        x = [x0]
        qr = lambda p: torch.view_as_real(self.q(p))

        for mu_lam in self.lambdas:
            m, l = self.softplus(mu_lam)
            # with torch.enable_grad():
            # _, Jf = functorch.vjp(qr, p)
            # # qpc = torch.view_as_complex(qp)
            # qpc = self.q(p)
            # Ahr = torch.view_as_real(enc.AHA(qpc) - Ahk)
            # Jr = Jf(Ahr, create_graph=True)[0]
            # dnet = self.xnet(p)
            # #dp = #self.reg_dp(m * (Jr + l * dnet))
            # dp = torch.tanh(m*(Jr+l*dnet))

            # qpc = self.q(p)
            qp, Jf = functorch.vjp(qr, p)
            qpc = torch.view_as_complex(qp)
            Ahr = torch.view_as_real(enc.AHA(qpc) - Ahk)
            Jr = Jf(Ahr, create_graph=True)[0]
            dnet = self.xnet(p)

            dp = self.reg_dp(m * (Jr + l * dnet))

            pneu = p - dp
            #

            # p.requires_grad_(True)
            # qp = torch.view_as_real(self.q(p))
            # Ahr = torch.view_as_real(enc.AHA(torch.view_as_complex(qp)) - Ahk)
            # Jr = torch.autograd.grad(qp, p, Ahr, create_graph=True, retain_graph=True)[0]
            # pcnn = self.xnet(p)
            # pneu = p - (m * (l * pcnn + Jr))
            # pcnn_reg = self.xnet(p.detach())
            xreg.append(p - m * l * dnet)
            p = pneu
            y.append(qpc)
            x.append(p)
        return (x, y, xreg, yreg)

    @property
    def device(self):
        return next(self.parameters()).device


class dopamine(Solution):
    def __init__(
        self,
        Problem,
        initial_lambdas: Tuple[float, float] = (0.05, 0.5),
        ylayer: int = 5,
        yinitial_filters: int = 64,
        iterations: int = 10,
        xlayer: int = 5,
        xinitial_filters: int = 64,
        lr_net=1e-4,
        lr_lam=1e-4,
        warmup_length: int = 500,
        pretrain_length: int = 1,
        min_lr: float = 1e-6,
        prev_loss_weight: float = 0.0,
        parameters=(2, 1),
        weight_decay: float = 1e-2,
        y0iterations: int = 2,
        tanh_bound: Union[float, bool] = True,
        maplossweight: float = 0.2,
    ):
        super().__init__(Problem)

        if parameters is None:
            parameters = [1]
        else:
            parameters = list(parameters)
        while sum(parameters) < self.q.nIn:
            parameters.append(1)

        self.net = _dopamine(
            self.q,
            initial_lambdas,
            iterations=iterations,
            n_data=self.q.nOut,
            parameters=parameters,
            xlayer=xlayer,
            xfilters=xinitial_filters,
            ylayer=ylayer,
            yfilters=yinitial_filters,
            y0iterations=y0iterations,
            tanh_bound=tanh_bound,
        )
        self.additional_info = dict(
            number_parameters=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.xnet.parameters())]) + sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.ynet.parameters())]),
            number_lambdas=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.lambdas.parameters())]),
        )
        self.bad = 0

    def forward(self, *x):
        return self.net(*x)

    def xloss(self, gtx, gty, pred, mask):
        mask = mask + 1e-1
        loss = nn.functional.mse_loss(mask * gtx, mask * pred[0][-1])
        ploss = [nn.functional.mse_loss(mask * gtx, mask * p).item() for p in pred[0]]
        for i, p in enumerate(ploss):
            self.log(str(i), p, prog_bar=True)
        if self.hparams.prev_loss_weight:
            loss = loss + self.hparams.prev_loss_weight * sum([nn.functional.mse_loss(mask * gtx, mask * p) for p in pred[0][:-1]])
        return loss

    def maploss(self, gtx, gty, pred, mask):
        loss = nn.functional.mse_loss(gtx, pred[0][0])
        return loss

    def loss(self, gt, pred, mask):
        return self.xloss(gt, None, pred, mask)

    def step(self, batch, opt, sched, pre=False):
        if self.bad > 20:
            raise ValueError("to many bad steps in series!")

        if pre:
            lossfunction = lambda *x: self.maploss(*x)
        else:
            lossfunction = lambda *x: self.xloss(*x) + self.hparams.maplossweight * self.maploss(*x)
        opt.zero_grad()
        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)
        k = enc.A(y)
        k += enc.mask(noise)
        p = self(k, enc)
        loss = lossfunction(x, y, p, mask)
        if (loss > 0.1 and ((self.trainer.global_step - self.batchesPre) > 20)) or not torch.isfinite(loss):
            print("Loss too high, no backprop done", loss)
            sched.step()
            self.bad += 1
            return loss

        self.manual_backward(loss)
        try:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0, error_if_nonfinite=True)
        except Exception as e:
            print("grad error:", e)
            for n, p in self.net.named_parameters():
                if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                    print(n, p.grad.ravel()[:5])
                    self.bad += 1
                    sched.step()
                    return loss

        opt.step()
        sched.step()
        self.bad = 0
        return loss

    def pretrain_step(self, batch, opt, sched, batch_idx):
        return self.step(batch, opt, sched, True)

    def train_step(self, batch, opt, sched, batch_idx):
        return self.step(batch, opt, sched, False)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            [
                {"params": self.net.xnet.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay},
                {"params": self.net.ynet.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay},
                {"params": self.net.lambdas.parameters(), "lr": self.hparams.lr_lam, "weight_decay": self.hparams.weight_decay},
            ]
        )
        optimPre = torch.optim.AdamW([{"params": self.net.ynet.parameters(), "lr": 10 * self.hparams.lr_net}])
        stepsTotal = self.trainer.fit_loop.max_epochs
        self.trainer.fit_loop.max_epochs = stepsTotal - self.hparams.pretrain_length
        batches = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = self.hparams.pretrain_length
        batchesPre = self.trainer.estimated_stepping_batches
        self.batchesPre = batchesPre
        self.trainer.fit_loop.max_epochs = stepsTotal
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
