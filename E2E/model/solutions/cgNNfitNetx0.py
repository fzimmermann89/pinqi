import torch
from torch import nn
from typing import Tuple, Optional, Literal
from einops import rearrange

from .cg import CGO, cg
from .bfgs import BFGS
from ...net.layers import ScaledTanh, IterationEmbedding
from ...net import Unet
from .base import Solution
from ...util import WarmupLR, softplus_inv
from ...util.bind import bind, invbind


class _cgNNfitNetx0(torch.nn.Module):
    def __init__(
        self,
        q,
        n_data: int = 10,
        n_param: int = 2,
        initial_lambdas: Tuple[float, ...] = (1.0,),
        iterations: int = 1,
        tag: bool = True,
        nettype: Literal["channels", "seperate", "3d", "2.5d"] = "seperate",
        netargs: Optional[dict] = None,
        layer: int = 3,
        res: bool = False,
        initial_filters: int = 16,
        feature_growth: Tuple[float, ...] = (2, 2, 1.5, 1),
        y0iterations: int = 2,
        x0layer: int = 2,
        x0initial_filters: int = 16,
        x0feature_growth: Tuple[float, ...] = (2.0, 2.0, 1.0, 1.0),
        x0netargs=None,
    ):
        super().__init__()
        if netargs is None:
            netargs = {}
        if x0netargs is None:
            x0netargs = {}

        self.softplus = nn.Softplus(beta=5.0)
        self.tagNetInput = tag
        self.iterations = iterations
        feature_growth = list(feature_growth) + (layer - len(feature_growth) + 2) * [1]

        if tag:
            if iterations < 2 and nettype != "seperate":
                raise ValueError("tagging only useful if multiple iteratations or seperated net")
            self.iterationEmbedding = IterationEmbedding(max_n=1.0, dim1=32, dim2=64)

        self.nettype = nettype
        shared = dict(filters=initial_filters, feature_growth=lambda i: feature_growth[i], layer=layer, **netargs)
        if nettype == "channels":
            self.net = Unet(2, channels_in=2 * n_data + tag, channels_out=2 * n_data, **shared)
        elif nettype == "3d":
            self.net = Unet(3, channels_in=2 + tag, channels_out=2, **shared)
        elif nettype == "2.5d":
            self.net = Unet(2.5, channels_in=2 + tag, channels_out=2, **shared)
        elif nettype == "seperate":
            self.net = Unet(2, channels_in=2 + tag, channels_out=2, **shared)
        else:
            raise ValueError(f"unknown nettype {self.nettype}")

        self.xnet = Unet(2, channels_in=2 * n_data, channels_out=n_param, x0filters=x0initial_filters, feature_growth=lambda i: x0feature_growth[i], x0layer=x0layer, **x0netargs)

        self.lam = nn.Parameter(torch.as_tensor(initial_lambdas, dtype=torch.float32))
        self.q = q
        self.n_data = n_data
        self.lambdas = nn.ParameterList(nn.Parameter(softplus_inv(torch.as_tensor(l, dtype=torch.float32)), self.softplus.beta) for l in initial_lambdas)
        self.y0iterations = y0iterations
        self.bad = 0
        self.res = res

    def get_lambda(self, iteration):
        return self.softplus(self.lambdas[min(iteration, len(self.lambdas) - 1)])

    def get_x0(self, y):
        bounds = self.q.bounds
        x0 = invbind(bounds, self.q.initial_values.expand(y.shape[0], -1, *y.shape[-2:]))
        x0 = x0 + self.xnet(y)
        x0 = bind(bounds, x0)
        return x0

    def forward(self, k, enc, e2e=True):
        Ahk = enc.AH(k)
        x, y, xreg, yreg = [], [], [], []

        if self.y0iterations > 0:
            with torch.no_grad():
                y.append(cg(enc.AHA, Ahk, Ahk, maxiter=self.y0iterations, dims=(-1, -2, -3)))
        else:
            y.append(Ahk)

        for iteration in range(self.iterations):
            yreg.append(self.apply_net(y[-1]), iteration)
            y.append(self.solve_problem1(enc, Ahk, y0=yreg[-1], yreg=yreg[-1], l=self.get_lambda(iteration)))

        x0 = self.get_x0(y[-1] if e2e else y[0])
        x = [x0]
        xreg = [x0]
        x.append(self.solve_problem2(self.q, y[-1], x[-1]))
        return (x, y, xreg, yreg)

    def apply_net(self, y, iteration=0.0):
        yin = torch.view_as_real(y)

        if self.nettype == "seperate":
            yin = rearrange(yin, "b c h w r -> (b c) r h w")
        elif self.nettype == "channels":
            yin = rearrange(yin, "b c h w r -> b (c r) h w")
        elif self.nettype == "3d" or self.nettype == "2.5d":
            yin = rearrange(yin, "b c h w r -> b r c h w")
        else:
            raise ValueError(f"unknown nettype {self.nettype}")

        if self.tag:

            tag = (iteration) / self.iterations
            tag = torch.as_tensor(tag).reshape(-1)

            if self.nettype == "seperate":
                channel_emb = torch.broadcast_to(torch.linspace(0.0, 1 / self.iterations, y.shape[1] + 1)[:-1], (y.shape[0], -1)).ravel()
                tag = torch.broadcast_to(tag.unsqueeze(-1), (y.shape[0], y.shape[1])).ravel()
                tag = channel_emb + tag

            tag = tag.to(device=yin.device)
            embiter = tag
            tag = torch.broadcast_to(tag[:, None, None, None], (yin.shape[0], 1, *yin.shape[2:]))
            tagged = torch.cat((yin, tag), 1)
            embiter = self.iterationEmbedding(embiter)
        else:
            tagged = yin
            embiter = None

        yout = self.net(tagged, embiter)

        if self.nettype == "seperate":
            yout = rearrange(yout, "(b c) r h w ->b c h w r", r=2, b=len(y))
        elif self.nettype == "channels":
            yout = rearrange(yout, "b (c r) h w -> b c h w r", r=2, b=len(y))
        elif self.nettype == "3d" or self.nettype == "2.5d":
            yout = rearrange(yout, "b r c h w -> b c h w r", r=2, b=len(y))
        else:
            raise ValueError(f"unknown nettype {self.nettype}")

        yout = torch.view_as_complex(yout.contiguous())
        if self.res:
            yout = y + yout
        return yout

    @staticmethod
    def solve_problem1(enc, Ahk, y0, yreg, l):
        r = CGO.apply(enc.AHA, Ahk, yreg, y0, l, (-1, -2, -3))
        return r

    @staticmethod
    def solve_problem2(q, y):
        x0 = q.initial_values.expand(y.shape[0], -1, *y.shape[-2:]).clone()
        r = BFGS.apply(lambda x: torch.view_as_real(q(x)), torch.view_as_real(y), x0, torch.zeros(1, device=x0.device), x0, q.bounds)
        return r

    @property
    def device(self):
        return next(self.parameters()).device


class cgNNfitNetx0(Solution):
    def __init__(
        self,
        Problem,
        initial_lambdas: Tuple[float, ...] = (1.0,),
        iterations: int = 1,
        layer: int = 3,
        initial_filters: int = 16,
        res: bool = True,
        feature_growth: Tuple[float, ...] = (2, 2, 1.5, 1),
        tag: bool = True,
        nettype: Literal["channels", "seperate", "3d", "2.5d"] = "seperate",
        netargs: Optional[dict] = None,
        lr_net=1e-3,
        lr_lam=1e-2,
        warmup_length: int = 1000,
        pretrain_length: int = 1,
        min_lr: float = 1e-5,
        outsidemaskstrength: float = 1e-3,
        weight_decay: float = 1e-2,
        y0iterations: int = 2,
        x0layer: int = 2,
        x0initial_filters: int = 16,
        x0feature_growth: Tuple[float, ...] = (2.0, 2.0, 1.0, 1.0),
        x0netargs=None,
    ):
        super().__init__(Problem)
        self.net = _cgNNfitNetx0(
            self.q,
            n_data=self.q.nOut,
            n_param=self.q.nIn,
            initial_lambdas=initial_lambdas,
            iterations=iterations,
            tag=tag,
            nettype=nettype,
            netargs=netargs,
            layer=layer,
            initial_filters=initial_filters,
            res=res,
            feature_growth=feature_growth,
            y0iterations=y0iterations,
            x0layer=x0layer,
            x0initial_filters=x0initial_filters,
            x0feature_growth=x0feature_growth,
            x0netargs=x0netargs,
        )
        self.additional_info = dict(
            number_parameters=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.net.parameters())]) + sum([p.numel() for p in filter(lambda p: p.requires_grad, self.xnet.net.parameters())]),
            number_lambdas=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.lambdas.parameters())]),
        )

    def forward(self, *x):
        return self.net(*x)

    def e2eloss(self, gtx, gty, pred, mask):
        mask = mask + self.hparams.outsidemaskstrength
        loss = nn.functional.mse_loss(mask * gtx, mask * pred[0][-1])
        return loss

    def nnloss(self, gtx, gty, pred, mask):
        loss = nn.functional.mse_loss(torch.view_as_real(mask * (gty + 0j)), torch.view_as_real(mask * pred[1][-1]))
        return loss

    def x0loss(self, gtx, gty, pred, mask):
        mask = mask + self.hparams.outsidemaskstrength
        loss = nn.functional.mse_loss(mask * gtx, mask * pred[2][-1])
        return loss

    def loss(self, gt, pred, mask):
        return self.e2eloss(gt, None, pred, mask)

    def step(self, batch, opt, sched, lossfunction):
        if self.bad > 10:
            raise ValueError("to many bad steps in series!")
        opt.zero_grad()
        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)
        k = enc.A(y)
        k += enc.mask(noise)
        p = self(k, enc)
        loss = lossfunction(x, y, p, mask)

        if loss > 5 and self.trainer.global_step > 2:
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
                if p.grad is None:
                    continue
                if not torch.all(torch.isfinite(p.grad)):
                    print(n, p.grad.ravel()[:5])
                    p.grad.fill_(0.0)
            self.bad += 1
            sched.step()
            return loss
        opt.step()
        sched.step()
        self.bad = 0
        return loss

    def pretrain_step(self, batch, opt, sched, batch_idx):
        return self.step(batch, opt, sched, lambda *x: self.nnloss(*x) + self.x0loss(*x), e2e=False)

    def train_step(self, batch, opt, sched, batch_idx):
        return self.step(batch, opt, sched, lambda *x: self.e2eloss(*x) + self.x0loss(*x), e2e=True)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            [
                {"params": self.net.net.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay},
                {"params": self.net.xnet.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay},
                {"params": self.net.lambdas.parameters(), "lr": self.hparams.lr_lam, "weight_decay": 1e-2 * self.hparams.weight_decay},
            ]
        )
        optimPre = torch.optim.AdamW(
            [
                {"params": self.net.net.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay},
                {"params": self.net.xnet.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay},
                {"params": self.net.lambdas.parameters(), "lr": self.hparams.lr_lam, "weight_decay": 1e-2 * self.hparams.weight_decay},
            ]
        )

        stepsTotal = self.trainer.fit_loop.max_epochs
        self.trainer.fit_loop.max_epochs = stepsTotal - self.hparams.pretrain_length
        batches = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = self.hparams.pretrain_length
        batchesPre = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = stepsTotal

        sched = WarmupLR(
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, batches - self.hparams.warmup_length, eta_min=self.hparams.min_lr, verbose=False),
            self.hparams.min_lr,
            self.hparams.warmup_length,
        )

        warmupPre = self.hparams.warmup_length  # max(int(self.hparams.warmup_length / batches * batchesPre), 100)
        schedPre = WarmupLR(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimPre, batchesPre - warmupPre, eta_min=self.hparams.min_lr, verbose=False),
            self.hparams.min_lr,
            warmupPre,
        )
        return [optim, optimPre], [sched, schedPre]
