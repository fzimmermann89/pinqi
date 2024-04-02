import torch
import pytorch_lightning as pl
from torch import nn
from typing import Tuple, Optional
import numpy as np
import functorch

from .cg import cg
from ...net import Unet
from ...net.blocks import CBlock
from .base import Solution
from ...util import WarmupLR, softplus_inv
from ...net.layers import IterationEmbedding
from ...util.bind import bind, invbind


def _applyJ(J, x):
    return torch.einsum("b...oci,bi...->bo...c", J, x)


def _applyJT(J, x):
    return torch.einsum("b...oci,bo...c->bi...", J, x)


class _unrolledGN(torch.nn.Module):
    @staticmethod
    @torch.enable_grad()
    def Jacobian(f, x0):
        J = functorch.vmap(functorch.vmap(functorch.jacfwd(lambda x: f(x[None, ..., None, None])[0, :, 0, 0, ...]), -1), 0)(x0.flatten(start_dim=2))
        J = J.reshape(J.shape[0], *x0.shape[2:], *J.shape[2:])
        return J

    class solve_p(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Dq, EHE, diff, lreg, llm, deltaNet):
            # Dq: Jacobian of Q, EHE: R2R of AHA, diff: EHE(q(xn))-Ahk, deltaNet: xcnn = xn+deltaNet
            def H(v):
                Dqv = _applyJ(Dq, v)
                EHE_Dqv = EHE(Dqv)
                DqTAHADqv = _applyJT(Dq, EHE_Dqv)
                ret = DqTAHADqv + (lreg + llm) * v
                if torch.any(torch.isnan(ret)):
                    print("H nan")
                    test = [Dq, Dqv, EHE_Dqv, DqTAHADqv, lreg, llm, v, ret]
                    print("[Dq,Dqv,EHE_Dqv,DqTAHADqv,lreg,llm,v,ret]", [(torch.isnan(i).any(), torch.isfinite(i).all()) for i in test])

                return ret

            b = -_applyJT(Dq, diff) + lreg * deltaNet
            p = cg(H, b, dims=(-1, -2, -3), x0=b / (llm + lreg), tol=1e-12)  # TODO check dims
            ctx.save_for_backward(p, deltaNet, lreg, llm, Dq, diff)
            ctx.H = H
            ctx.EHE = EHE
            return p

        @staticmethod
        def backward(ctx, grad_output):
            p, deltaNet, lreg, llm, Dq, diff = ctx.saved_tensors

            d = -cg(ctx.H, grad_output, dims=(-1, -2, -3), x0=grad_output, tol=1e-12)  # TODO check dims

            ddiff = _applyJ(Dq, d) if ctx.needs_input_grad[2] else None
            dlreg = torch.sum(d * (p - deltaNet), (-1, -2, -3)) if ctx.needs_input_grad[3] else None  # TODO check dims
            dllm = torch.sum(d * p, (-1, -2, -3)) if ctx.needs_input_grad[4] else None  # TODO check dims
            ddeltaNet = -d * lreg if ctx.needs_input_grad[5] else None

            if ctx.needs_input_grad[0]:
                # first, i.e the (...)Jx derivative of H
                Jd = ddiff if ddiff is not None else _applyJ(Dq, d)
                t1 = ctx.EHE(Jd)
                t1 = torch.einsum("bixy,boxyc->bxyoci", p, t1)

                # second, i.e the j (ahajx) derivative of H
                t2 = ctx.EHE(_applyJ(Dq, p))
                # ..inluding the j derivative of b which is einsum("boxyc,bixy->bxyoci", diff, d)
                t2 = torch.einsum("boxyc,bixy->bxyoci", t2 + diff, d)
                dDq = t1 + t2
            else:
                dDq = None
            dAHA = None
            return dDq, dAHA, ddiff, dlreg, dllm, ddeltaNet

    def __init__(
        self,
        q,
        initial_lambdas: Tuple[Tuple[float, float, float], ...] = ((0.1, 1.0, 1.0),),
        iterations: int = 4,
        n_data: int = 10,
        n_param: int = 3,
        tag: bool = True,
        shiftscale: bool = True,
        input_data: bool = False,
        input_param: bool = True,
        input_diff: bool = False,
        input_x0: bool = False,
        initial_filters: int = 32,
        feature_growth: Tuple[float, ...] = (2, 1.5, 1.5),
        layer: int = 4,
        initial_net_shape: Tuple[int, ...] = (32, 64, 64),
        y0iterations: int = 0,
    ):
        super().__init__()
        if not (input_data or input_param or input_diff or input_x0):
            raise ValueError("At least one of input param, data, diff, x0 shall be true to have an image dependent input for the net")
        if len(initial_lambdas) > 1:
            if iterations is None:
                iterations = len(initial_lambdas)
            elif len(initial_lambdas) != iterations:
                raise ValueError("length missmatch between initial_lambdas and iterations")
        feature_growth = list(feature_growth) + (layer - len(feature_growth) + 2) * [1]
        if initial_net_shape:
            self.initial_net = nn.Sequential(
                CBlock((2 * n_data, *initial_net_shape), kernel_size=3, activation=torch.nn.SiLU),
                nn.Conv2d(initial_net_shape[-1], n_param, kernel_size=1),
            )
            with torch.no_grad():
                self.initial_net[-1].bias.fill_(0.0)
                self.initial_net[-1].weight.normal_(mean=0, std=0.01)
        else:
            self.initial_net = False
        self.softplus = nn.Softplus(beta=5.0)
        self.netinput_tag = tag
        self.netinput_data = input_data
        self.netinput_diff = input_diff
        self.netinput_param = input_param
        self.netinput_x0 = input_x0

        channels_in = tag + 2 * n_data * input_data + 2 * n_data * input_diff + input_param * n_param + input_x0 * n_param
        self.net = Unet(
            2,
            channels_in=channels_in,
            channels_out=n_param,
            filters=initial_filters,
            feature_growth=lambda i: feature_growth[i],
            layer=layer,
            emb_dim=64 if shiftscale else 0,
        )
        self.lambdas = nn.ModuleList(nn.ParameterList(nn.Parameter(softplus_inv(torch.as_tensor(l, dtype=torch.float64), self.softplus.beta)) for l in ls) for ls in initial_lambdas)
        self.q = q
        self.iterations = iterations
        self.iterationEmbedding = IterationEmbedding(max_n=iterations, dim1=32, dim2=64) if shiftscale else None
        self.y0iterations = y0iterations

    def get_lambda(self, iteration, subproblem):
        return self.softplus(self.lambdas[min(iteration, len(self.lambdas) - 1)][subproblem])

    def prepareNetInput(self, x, diff, EHk, iteration, x0):
        inputs = []
        if self.netinput_tag:
            tag = (iteration / self.iterations) * torch.ones(x.shape[0], 1, *x.shape[-2:], device=x.device)
            inputs.append(tag)
        if self.netinput_data:
            inputs.append(self.join_channels(EHk))
        if self.netinput_x0:
            inputs.append(x0)
        if self.netinput_diff:
            inputs.append(self.join_channels(diff))
        if self.netinput_param:
            inputs.append(x)
        return torch.cat(inputs, 1)

    def forward(self, k, enc):
        Ahk = enc.AH(k)
        EHk = torch.view_as_real(Ahk)
        EHE = lambda x: torch.view_as_real(enc.AHA(torch.view_as_complex(x)))
        bounds = self.q.bounds
        qr = lambda x: torch.view_as_real(self.q(bind(x, bounds)) + 0j)
        x0 = self.q.initial_values.expand(k.shape[0], -1, *k.shape[-2:])
        xreg, yreg, y = [], [], []
        if self.initial_net:
            if self.y0iterations > 0:
                with torch.no_grad():
                    y0 = torch.view_as_real(cg(enc.AHA, Ahk, Ahk, maxiter=self.y0iterations, dims=(-1, -2, -3)))
            else:
                y0 = EHk
            x0 = x0 + self.initial_net(self.join_channels(y0)) * self.q.scale

        x0 = invbind(x0, bounds)
        x, xreg = [x0], []
        embed = None
        for iteration in range(self.iterations):
            lreg, llm, alpha = self.get_lambda(iteration, 0), self.get_lambda(iteration, 1), self.get_lambda(iteration, 2)  # just gets the softplus(parameter)
            qx = qr(x[-1])
            Dq = self.Jacobian(qr, x[-1])
            EHEqx = EHE(qx)
            diff = EHEqx - EHk
            netinput = self.prepareNetInput(x[-1], diff, EHk, iteration, x0)
            if self.iterationEmbedding is not None:
                embed = self.iterationEmbedding(torch.as_tensor(iteration, device=netinput.device)[..., None].ravel())
            p_net = self.net(netinput, embed)
            pk = self.solve_p.apply(Dq, EHE, diff, lreg, llm, p_net)
            x.append(x[-1] + alpha * pk)
            xreg.append(x[-1] + p_net)
        xreg = [bind(i, bounds) for i in xreg]
        x = [bind(i, bounds) for i in x]
        return (x, y, xreg, yreg)

    @staticmethod
    def join_channels(y):
        return y.moveaxis(-1, 1).reshape(y.shape[0], -1, *y.shape[2:-1])

    @property
    def device(self):
        return next(self.parameters()).device


class unrolledGN(Solution):
    def __init__(
        self,
        Problem,
        initial_lambdas: Tuple[Tuple[float, float, float], ...] = ((0.1, 1.0, 1.0),),
        iterations: int = 3,
        tag: bool = True,
        shiftscale: bool = True,
        input_data: bool = False,
        input_param: bool = True,
        input_diff: bool = False,
        input_x0: bool = False,
        lr_net=3e-4,
        lr_lam=1e-3,
        outsidemaskstrength: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_length: int = 1000,
        pretrain_length: int = 5,
        min_lr: float = 1e-5,
        loss_prev_weights: Optional[Tuple[float, ...]] = None,
        initial_filters: int = 32,
        feature_growth: Tuple[float, ...] = (2, 1.5, 1.5, 1),
        layer: int = 4,
        initial_net_shape: Tuple[int, ...] = (32, 64, 64),
        y0iterations: int = 2,
    ):
        super().__init__(Problem)
        self.net = _unrolledGN(
            self.q,
            initial_lambdas,
            iterations,
            self.q.nOut,
            self.q.nIn,
            tag,
            shiftscale,
            input_data,
            input_param,
            input_diff,
            input_x0,
            initial_filters,
            feature_growth,
            layer,
            initial_net_shape,
        )
        self.additional_info = dict(
            number_parameters=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.net.parameters())]),
            number_lambdas=sum([p.numel() for p in filter(lambda p: p.requires_grad, self.net.lambdas.parameters())]),
        )
        self.bad = 0

    def forward(self, *x):
        return self.net(*x)

    def loss(self, gt, pred, mask):
        mask = mask.float() + self.hparams.outsidemaskstrength
        gtm = gt * mask

        loss = nn.functional.mse_loss(gtm, pred[0][-1] * mask)
        if self.hparams.loss_prev_weights is not None:
            weights = (
                self.hparams.loss_prev_weights[0],
                # *((self.hparams.iterations - len(self.hparams.loss_prev_weights)) * [0]),
                *(max(self.net.iterations - len(self.hparams.loss_prev_weights) - 1, 0) * [0]),
                *self.hparams.loss_prev_weights[1:],
            )
            for i, w in enumerate(weights):
                if w:
                    loss = loss + w * nn.functional.mse_loss(gtm, pred[0][i] * mask)
        return loss

    def pretrain_step(self, batch, opt, sched, batch_idx):
        prev_weights, self.hparams.loss_prev_weights = self.hparams.loss_prev_weights, (1.0 if self.net.initial_net else 0.0, 0.5)
        iterations, self.net.iterations = self.net.iterations, 2
        loss = self.train_step(batch, opt, sched)
        self.net.iterations, self.hparams.loss_prev_weights = iterations, prev_weights
        return loss

    def train_step(self, batch, opt, sched, batch_idx):
        if self.bad > 10:
            raise ValueError("to many bad steps!")
        opt.zero_grad()
        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)
        k = enc.A(y)
        k += enc.mask(noise)
        p = self(k, enc)
        loss = self.loss(x, p, mask)
        if loss > 10:
            if self.trainer.global_step > 5:
                print("Loss to high, no backprop done", loss)
                self.bad += 1
                sched.step()
                return loss
            else:
                print("loss high", loss)
        self.manual_backward(loss)

        try:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5, error_if_nonfinite=True)
        except Exception as e:
            print("loss", loss)
            print("grad error:", e)
            for n, p in self.named_parameters():
                if not torch.all(torch.isfinite(p.grad)):
                    print(n, p.grad.ravel()[:5])
                    p.grad.fill_(0.0)
            sched.step()
            self.bad += 1
            return loss
        opt.step()
        sched.step()
        self.bad = 0
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            [{"params": self.net.net.parameters(), "lr": self.hparams.lr_net, "weight_decay": self.hparams.weight_decay}, {"params": self.net.lambdas.parameters(), "lr": self.hparams.lr_lam, "weight_decay": 0.1 * self.hparams.weight_decay}]
            + [{"params": self.net.initial_net.parameters(), "lr": 0.1 * self.hparams.lr_net, "weight_decay": min(1e-2, 10 * self.hparams.weight_decay)}]
            if self.net.initial_net
            else [] + [{"params": self.net.iterationEmbedding.parameters(), "lr": self.hparams.lr_net, "weight_decay": 1e-2}]
            if self.iterationEmbedding
            else []
        )
        optimPre = torch.optim.AdamW(
            [
                {"params": self.net.net.parameters(), "lr": 0.1 * self.hparams.lr_net, "weight_decay": 1e-4},
                {"params": self.net.lambdas.parameters(), "lr": 0.03 * self.hparams.lr_lam},
            ]
            + [{"params": self.net.initial_net.parameters(), "lr": 1.0 * self.hparams.lr_net, "weight_decay": 1e-3}]
            if self.net.initial_net
            else []
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

        warmupPre = max(int(self.hparams.warmup_length / batches * batchesPre), 100)
        schedPre = WarmupLR(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimPre, batchesPre - warmupPre, eta_min=self.hparams.min_lr, verbose=False),
            self.hparams.min_lr,
            warmupPre,
        )
        return [optim, optimPre], [sched, schedPre]
