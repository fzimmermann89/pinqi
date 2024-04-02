import torch
from torch import nn, Tensor
from typing import Tuple, Optional, Literal
from functools import partial
from einops import rearrange

from .cg import CGO, cg
from ...net import Unet
from ...net.layers import IterationEmbedding
from .base import Solution
from ...util import WarmupLR, softplus_inv


class _cgNN2(torch.nn.Module):
    def __init__(
        self,
        q,
        initial_lambdas: Tuple[float, ...] = (0.1,),
        n_data: int = 10,
        n_param: int = 2,
        ytag: bool = False,
        ylayer: int = 4,
        yres: bool = True,
        yinitial_filters: int = 32,
        yfeature_growth: Tuple[float, ...] = (2, 2, 1.5, 1),
        yiterations: int = 1,
        ynettype: Literal["channels", "seperate", "3d", "2.5d"] = "seperate",
        ynetargs: Optional[dict] = None,
        xlayer: int = 4,
        xinitial_filters: int = 32,
        xfeature_growth: Tuple[float, ...] = (2, 2, 1.5, 1),
        xnetargs: Optional[dict] = None,
        rescale: bool = True,
        y0iterations: int = 0,
    ):
        super().__init__()
        self.softplus = nn.Softplus(beta=5.0)
        self.tagNetInput = ytag
        self.iterations = yiterations
        yfeature_growth = list(yfeature_growth) + (ylayer - len(yfeature_growth) + 2) * [1]
        xfeature_growth = list(xfeature_growth) + (xlayer - len(xfeature_growth) + 2) * [1]
        if ynetargs is None:
            ynetargs = {}
        if xnetargs is None:
            xnetargs = {}

        self.ynettype = ynettype
        shared = dict(filters=yinitial_filters, feature_growth=lambda i: yfeature_growth[i], layer=ylayer, **ynetargs)
        if ynettype == "channels":
            self.ynet = Unet(2, channels_in=2 * n_data + ytag, channels_out=2 * n_data, **shared)
        elif ynettype == "3d":
            self.ynet = Unet(3, channels_in=2 + ytag, channels_out=2, **shared)
        elif ynettype == "2.5d":
            self.ynet = Unet(2.5, channels_in=2 + ytag, channels_out=2, **shared)
        elif ynettype == "seperate":
            self.ynet = Unet(2, channels_in=2 + ytag, channels_out=2, **shared)
        else:
            raise ValueError(f"unknown ynettype {self.ynettype}")

        self.xnet = torch.nn.Sequential(Unet(2, channels_in=2 * n_data, channels_out=n_param, filters=xinitial_filters, feature_growth=lambda i: xfeature_growth[i], layer=xlayer, **xnetargs))
        self.lambdas = nn.ParameterList(nn.Parameter(softplus_inv(torch.as_tensor(l, dtype=torch.float32), self.softplus.beta)) for l in initial_lambdas)

        if ytag:
            if yiterations < 2 and ynettype != "seperate":
                raise ValueError("tagging only useful if multiple iteratations or seperated net")
            self.iterationEmbedding = IterationEmbedding(max_n=1.0, dim1=32, dim2=64)

        self.ytag = ytag
        self.q = q
        self.n_data = n_data
        self.rescale = rescale
        self.y0iterations = y0iterations
        self.yres = yres
        self.yiterations = yiterations

    def get_lambda(self, iteration: int) -> Tensor:
        return self.softplus(self.lambdas[min(iteration, len(self.lambdas) - 1)])

    def forward(self, k: Tensor, enc, e2e: bool = True) -> Tensor:
        Ahk = enc.AH(k)
        x, y, xreg, yreg = [], [], [], []
        if self.y0iterations > 0:
            with torch.no_grad():
                y.append(cg(enc.AHA, Ahk, Ahk, maxiter=self.y0iterations, dims=(-1, -2, -3)))
        else:
            y.append(Ahk)

        for iteration in range(self.iterations):
            yreg.append(self.apply_ynet(y[-1], iteration))
            y.append(self.solve_problem1(enc, Ahk, y0=y[-1], yreg=yreg[-1], l=self.get_lambda(iteration)))

        finaly = self.reshape_c2r_y(y[-1])
        if not e2e:
            finaly = finaly.detach()
        xp = self.xnet(finaly)
        if self.rescale:
            xp = (torch.tanh(xp * 0.1) * 10 * self.q.scale) + self.q.initial_values

        x.append(xp)
        return (x, y, xreg, yreg)

    def apply_ynet(self, y, iteration=0.0):
        yin = torch.view_as_real(y)

        if self.ynettype == "seperate":
            yin = rearrange(yin, "b c h w r -> (b c) r h w")
        elif self.ynettype == "channels":
            yin = rearrange(yin, "b c h w r -> b (c r) h w")
        elif self.ynettype == "3d" or self.ynettype == "2.5d":
            yin = rearrange(yin, "b c h w r -> b r c h w")
        else:
            raise ValueError(f"unknown ynettype {self.ynettype}")

        if self.ytag:

            tag = (iteration) / self.yiterations
            tag = torch.as_tensor(tag).reshape(-1)

            if self.ynettype == "seperate":
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

        yout = self.ynet(tagged, embiter)

        if self.ynettype == "seperate":
            yout = rearrange(yout, "(b c) r h w ->b c h w r", r=2, b=len(y))
        elif self.ynettype == "channels":
            yout = rearrange(yout, "b (c r) h w -> b c h w r", r=2, b=len(y))
        elif self.ynettype == "3d" or self.ynettype == "2.5d":
            yout = rearrange(yout, "b r c h w -> b c h w r", r=2, b=len(y))
        else:
            raise ValueError(f"unknown ynettype {self.ynettype}")

        yout = torch.view_as_complex(yout.contiguous())
        if self.yres:
            yout = y + yout
        return yout

    @staticmethod
    def reshape_c2r_y(y: Tensor) -> Tensor:
        return torch.view_as_real(y).moveaxis(-1, 1).reshape(y.shape[0], -1, *y.shape[-2:])

    @staticmethod
    def solve_problem1(enc, Ahk: Tensor, y0: Tensor, yreg: Tensor, l: Tensor) -> Tensor:
        r = CGO.apply(enc.AHA, Ahk, yreg, y0, l, (-1, -2, -3))
        return r

    @property
    def device(self):
        return next(self.parameters()).device


class cgNN2(Solution):
    def __init__(
        self,
        Problem,
        initial_lambdas: Tuple[float, ...] = (1.0,),
        ytag: bool = True,
        ylayer: int = 3,
        yres: bool = True,
        yinitial_filters: int = 16,
        yfeature_growth: Tuple[float, ...] = (2, 2, 1.5, 1),
        yiterations: int = 1,
        ynettype: Literal["channels", "seperate", "3d", "2.5d"] = "seperate",
        ynetargs: Optional[dict] = None,
        xlayer: int = 3,
        xinitial_filters: int = 16,
        xfeature_growth: Tuple[float, ...] = (2, 2, 1.5, 1),
        xnetargs: Optional[dict] = None,
        lr_net: float = 1e-3,
        lr_lam: float = 1e-2,
        warmup_length: int = 1000,
        pretrain_length: int = 1,
        min_lr: float = 1e-5,
        e2e: bool = True,
        ylossweight: float = 0.2,
        outsidemaskstrength: float = 1e-3,
        weight_decay: float = 1e-4,
        rescale: bool = True,
        y0iterations: int = 2,
    ):
        super().__init__(Problem)
        self.net = _cgNN2(
            self.q,
            initial_lambdas,
            self.q.nOut,
            self.q.nIn,
            ytag,
            ylayer,
            yres,
            yinitial_filters,
            yfeature_growth,
            yiterations,
            ynettype,
            ynetargs,
            xlayer,
            xinitial_filters,
            xfeature_growth,
            xnetargs,
            rescale,
            y0iterations,
        )
        self.additional_info = dict(
            number_parameters=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.xnet.parameters())]) + sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.ynet.parameters())]),
            number_lambdas=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.lambdas.parameters())]),
        )

    def forward(self, *x):
        return self.net(*x)

    def xloss(self, gtx, gty, pred, mask):
        mask = mask + self.hparams.outsidemaskstrength
        loss = nn.functional.mse_loss(mask * gtx, mask * pred[0][-1])
        return loss

    def yloss(self, gtx, gty, pred, mask):
        mask = mask + max(1e-1, self.hparams.outsidemaskstrength)
        loss = nn.functional.mse_loss(torch.view_as_real(mask * gty), torch.view_as_real(mask * pred[1][-1]))
        return loss

    def loss(self, gt, pred, mask):
        return self.xloss(gt, None, pred, mask)

    def step(self, batch, opt, sched, pre=False):
        if pre:
            lossfunction = self.yloss
        elif self.hparams.e2e:
            lossfunction = lambda *x: self.xloss(*x) + self.hparams.ylossweight * self.yloss(*x)
        else:
            lossfunction = lambda *x: self.xloss(*x) + self.yloss(*x)
        opt.zero_grad()
        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)
        k = enc.A(y)
        k += enc.mask(noise)
        p = self(k, enc, self.hparams.e2e)
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
        optim = torch.optim.AdamW(
            [
                {"params": self.net.xnet.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay},
                {"params": self.net.ynet.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay},
                {"params": self.net.lambdas.parameters(), "lr": self.hparams.lr_lam, "weight_decay": 0.0},
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
