import torch
from torch import nn
from typing import Tuple, Optional, Literal, Union
from functools import partial
from einops import rearrange

from .cg import CGO, CGO2, cg
from .bfgs import BFGS
from ...net.layers import IterationEmbedding
from ...net import Unet
from .base import Solution
from ...util import WarmupLR, softplus_inv


class _pinqi(torch.nn.Module):
    def __init__(
        self,
        q,
        initial_lambdas: Union[
            Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float, float], ...]
        ] = ((1.0, 2.0),),
        iterations: int = 3,
        n_data: int = 10,
        n_param: int = 2,
        tag: bool = True,
        shiftscale: bool = True,
        iterateNet: bool = True,
        initial_filters: int = 32,
        feature_growth: Tuple[float, ...] = (2, 1.5, 1.5),
        layer: int = 4,
        netargs=None,
        ynettype: Literal["none", "channels", "seperate", "3d", "2.5d"] = "none",
        yinitial_filters: int = 32,
        ylayer: int = 3,
        yiterateNet: bool = True,
        ynetargs=None,
        y0iterations: int = 0,
        use_nonlinear_solver: bool = True,
    ):
        super().__init__()
        if len(initial_lambdas) > 1:
            if iterations is None:
                iterations = len(initial_lambdas)
            elif len(initial_lambdas) != iterations:
                raise ValueError(
                    "length missmatch between initial_lambdas and iterations"
                )
        ynet = yinitial_filters and ylayer and ynettype != "none"
        self.ir = netargs == "IR"
        xnet = not (netargs == False or self.ir)
        self.use_nonlinear_solver = use_nonlinear_solver
        if not (use_nonlinear_solver or xnet):
            raise ValueError("need xnet if not using nonlinear solver")
        if len(initial_lambdas[0]) != (xnet + ynet + self.ir + use_nonlinear_solver):
            raise ValueError("number of lambdas not matching!")

        feature_growth = feature_growth + (layer - len(feature_growth) + 2) * (1,)
        self.softplus = nn.Softplus(beta=5)
        self.tagNetInput = tag

        if ynetargs is None:
            ynetargs = {}
        if netargs is None:
            netargs = {}

        if xnet:
            self.net = Unet(
                2,
                channels_in=2 * n_data + tag,
                channels_out=n_param,
                filters=initial_filters,
                feature_growth=lambda i: feature_growth[i],
                layer=layer,
                emb_dim=64 if shiftscale else 0,
                **netargs,
            )
            torch.nn.init.normal_(self.net.last[0].weight, 0, 0.05)
            torch.nn.init.zeros_(self.net.last[0].bias)
        else:
            self.net = None
            self.xinit = torch.nn.Parameter(q.initial_values.clone())

        if ynet:
            shared = dict(
                filters=yinitial_filters,
                feature_growth=lambda i: feature_growth[i],
                layer=ylayer,
                **ynetargs,
            )
            self.ynettype = ynettype
            if ynettype == "channels":
                self.ynet = Unet(
                    2, channels_in=2 * n_data + tag, channels_out=2 * n_data, **shared
                )
            elif ynettype == "3d":
                self.ynet = Unet(3, channels_in=2 + tag, channels_out=2, **shared)
            elif ynettype == "2.5d":
                self.ynet = Unet(2.5, channels_in=2 + tag, channels_out=2, **shared)
            elif ynettype == "seperate":
                self.ynet = Unet(2, channels_in=2 + tag, channels_out=2, **shared)
            else:
                raise ValueError(f"unknown ynettype {self.ynettype}")
            torch.nn.init.zeros_(self.ynet.last[0].bias)
            torch.nn.init.normal_(self.ynet.last[0].weight, 0, 0.05)
        else:
            self.ynet = None

        self.lambdas = nn.ModuleList(
            nn.ParameterList(
                nn.Parameter(
                    softplus_inv(
                        torch.as_tensor(l, dtype=torch.float32), self.softplus.beta
                    )
                )
                for l in ls
            )
            for ls in initial_lambdas
        )
        self.q = q
        self.iterations = iterations
        self.iterateNet = iterateNet
        self.yiterateNet = yiterateNet
        self.y0iterations = y0iterations
        self.iterationEmbedding = (
            IterationEmbedding(max_n=1.0, dim1=32, dim2=64) if shiftscale else None
        )

    def _apply_ynet(self, y, rel_iteration):
        if self.ynet is None:
            raise ValueError("no ynet!")
        yin = torch.view_as_real(y)

        if self.ynettype == "seperate":
            yin = rearrange(yin, "b c h w r -> (b c) r h w")
        elif self.ynettype == "channels":
            yin = rearrange(yin, "b c h w r -> b (c r) h w")
        elif self.ynettype == "3d" or self.ynettype == "2.5d":
            yin = rearrange(yin, "b c h w r -> b r c h w")
        else:
            raise ValueError(f"unknown ynettype {self.ynettype}")

        if self.tagNetInput or (self.iterationEmbedding is not None):
            tag = torch.as_tensor(rel_iteration).reshape(-1)

            if self.ynettype == "seperate":
                channel_emb = (
                    torch.broadcast_to(
                        torch.linspace(0.0, 1 / self.iterations, y.shape[1] + 1)[:-1],
                        (y.shape[0], -1),
                    )
                    .ravel()
                    .to(y.device)
                )
                tag = torch.broadcast_to(
                    tag.unsqueeze(-1), (y.shape[0], y.shape[1])
                ).ravel()
                tag = channel_emb + tag

            tag = tag.to(device=yin.device)
            embiter = tag
            tag = tag.reshape((-1,) + (yin.ndim - tag.ndim) * (1,))
            tag = torch.broadcast_to(tag, (yin.shape[0], 1, *yin.shape[2:]))
            embiter = (
                None
                if self.iterationEmbedding is None
                else self.iterationEmbedding(embiter)
            )
            tagged = torch.cat((yin, tag), 1) if self.tagNetInput else yin
        else:
            embiter = None
            tagged = yin

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
        yout = y + yout
        return yout

    def get_lambda(
        self, iteration: int, subproblem: Literal["q", "x", "y"]
    ) -> torch.Tensor:
        if subproblem == "q":
            subproblem_index = 0
        elif subproblem == "x":
            if self.net is None and not self.ir:
                return torch.zeros(1, device=self.lambdas[0][0].device)
            elif not self.use_nonlinear_solver:
                return None
            else:
                subproblem_index = 1
        elif subproblem == "y":
            if self.ynet is None:
                return None
            elif self.net is None:
                subproblem_index = 1
            elif not self.use_nonlinear_solver:
                subproblem_index = 1
            else:
                subproblem_index = 2
        else:
            raise ValueError(f"unknown subproblem {subproblem}")
        return self.softplus(
            self.lambdas[min(iteration, len(self.lambdas) - 1)][subproblem_index]
        )  # type: ignore

    def forward(self, k, enc):
        Ahk = enc.AH(k)
        x, y, xreg, yreg, yreg2 = [], [], [], [], []

        y.append(Ahk)
        if self.y0iterations > 0:
            with torch.no_grad():
                y.append(
                    cg(enc.AHA, Ahk, Ahk, maxiter=self.y0iterations, dims=(-1, -2, -3))
                )
        if self.net is not None:
            x.append(self._apply_xnet(y[-1], 0))
            xreg.append(x[-1])
        else:
            shape = (y[-1].shape[0], self.xinit.shape[1], *y[-1].shape[2:])
            xinit = self.xinit.expand(*shape)
            x.append(xinit)

        for iteration in range(self.iterations):
            rel_iteration = (iteration + 1) / self.iterations
            yreg.append(self.q(x[-1]).to(torch.complex64))
            if self.ynet is not None:
                yreg2.append(
                    self._apply_ynet(y[-1], rel_iteration)
                    if self.yiterateNet or iteration == 0
                    else yreg2[-1]
                )

            y.append(
                self.solve_problem1(
                    enc,
                    Ahk,
                    y0=y[-1],
                    yreg1=yreg[-1],
                    l1=self.get_lambda(iteration, "q"),
                    yreg2=yreg2[-1] if len(yreg2) else None,
                    l2=self.get_lambda(iteration, "y"),
                )
            )

            if self.net is not None:
                xreg.append(
                    self._apply_xnet(y[-1], rel_iteration + 1)
                    if self.iterateNet
                    else xreg[-1]
                )
            if self.use_nonlinear_solver:
                x.append(
                    self.solve_problem2(
                        self.q,
                        y[-1],
                        x0=x[-1].detach(),
                        xreg=xreg[-1] if len(xreg) else x[-1],
                        l=self.get_lambda(iteration, "x"),
                    )
                )
            else:
                x.append(xreg[-1])
        if self.ynet:
            return (x, y, xreg, yreg, yreg2)
        else:
            return (x, y, xreg, yreg)

    def _apply_xnet(self, y, rel_iteration):
        tag = rel_iteration
        y = self.reshape_c2r_y(y)
        tagged = self.tag(y, tag)
        embiter = torch.as_tensor(tag, device=y.device).reshape(-1)
        embiter = (
            None
            if self.iterationEmbedding is None
            else self.iterationEmbedding(embiter)
        )
        x = (
            torch.tanh(self.net(tagged, embiter)) * (10 * self.q.scale)
            + self.q.initial_values
        )
        x[:, -1] = torch.nn.functional.softplus(x[:, -1] + 0.2, beta=5) - 0.2
        return x

    @staticmethod
    def reshape_c2r_y(y):
        return (
            torch.view_as_real(y).moveaxis(-1, 1).reshape(y.shape[0], -1, *y.shape[-2:])
        )

    def tag(self, y, i):
        if self.tagNetInput:
            t = torch.as_tensor(i, device=self.device)
            return torch.cat(
                (y, torch.broadcast_to(t, [y.shape[0], 1, *y.shape[2:]])), 1
            )
        else:
            return y

    @staticmethod
    def solve_problem1(enc, Ahk, y0, yreg1, l1, yreg2=None, l2=None):
        if yreg2 is not None and l2 is not None:
            r = CGO2.apply(enc.AHA, Ahk, yreg1, yreg2, y0, l1, l2, (-1, -2, -3))
        else:
            r = CGO.apply(enc.AHA, Ahk, yreg1, y0, l1, (-1, -2, -3))
        return r

    @staticmethod
    def solve_problem2(q, y, x0, xreg, l):
        r = BFGS.apply(
            lambda x: torch.view_as_real(q(x)),
            torch.view_as_real(y),
            xreg,
            l,
            x0.detach().clone(),
            q.bounds,
        )
        return r

    @property
    def device(self):
        return next(self.parameters()).device


class pinqi(Solution):
    def __init__(
        self,
        Problem,
        initial_lambdas: Union[
            Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float, float], ...]
        ] = ((1.0, 2.0),),
        iterations: int = 3,
        tag: bool = True,
        shiftscale: bool = False,
        iterateNet: bool = True,
        initial_filters: int = 32,
        feature_growth: Tuple[float, ...] = (2, 1.5, 1.5, 1),
        layer: int = 4,
        netargs: Union[dict, None, Literal[False], Literal["IR"]] = None,
        y0iterations: int = 2,
        yinitial_filters: int = 32,
        ylayer: int = 3,
        ynettype: Literal["none", "channels", "seperate", "3d", "2.5d"] = "none",
        yiterateNet: bool = True,
        ynetargs: Optional[dict] = None,
        use_nonlinear_solver: bool = True,
        loss: Literal["mse", "huber", "mae"] = "mse",
        outsidemaskstrength: float = 1e-2,
        lr_net: float = 1e-4,
        lr_lam: float = 3e-3,
        weight_decay: float = 1e-4,
        warmup_length: int = 200,
        pretrain_length: int = 1,
        min_lr: float = 1e-5,
        loss_prev_weights: Optional[Tuple[float, ...]] = (0.1, 0.1),
        debug: bool = False,
    ):
        super().__init__(Problem)
        self.net = _pinqi(
            self.q,
            initial_lambdas,
            iterations,
            self.q.nOut,
            self.q.nIn,
            tag,
            shiftscale,
            iterateNet,
            initial_filters,
            feature_growth,
            layer,
            netargs,
            ynettype,
            yinitial_filters,
            ylayer,
            yiterateNet,
            ynetargs,
            y0iterations,
            use_nonlinear_solver,
        )
        self.additional_info = dict(
            number_parameters=sum(
                [
                    p.numel()
                    for p in filter(lambda p: p.requires_grad, self.net.parameters())
                ]
            ),
            number_lambdas=sum(
                [
                    p.numel()
                    for p in filter(
                        lambda p: p.requires_grad, self.net.lambdas.parameters()
                    )
                ]
            ),
        )

        if loss == "huber":
            lossfunction = partial(torch.nn.functional.huber_loss, delta=0.1)
        elif loss == "mse":
            lossfunction = torch.nn.functional.mse_loss
        elif loss == "mae":
            lossfunction = torch.nn.functional.l1_loss
        else:
            raise NotImplementedError

        self.lossfun = lambda x, gt, mask: (
            lossfunction(x, gt, reduction="none") * mask
        ).mean()
        self.bad = 0

    def forward(self, *x):
        return self.net(*x)

    def loss(self, gt, pred, mask):
        nmask = mask.float() + self.hparams.outsidemaskstrength
        loss = self.lossfun(gt, pred[0][-1], nmask)
        if self.hparams.loss_prev_weights is not None:
            weights = (
                self.hparams.loss_prev_weights[0],
                *(
                    max(
                        self.hparams.iterations
                        - len(self.hparams.loss_prev_weights)
                        - 1,
                        0,
                    )
                    * [0]
                ),
                *self.hparams.loss_prev_weights[1:],
            )
            for i, w in enumerate(weights):
                if w:
                    loss = loss + w * self.lossfun(gt, pred[0][i], nmask)
        return loss

    def pretrain_step(self, batch, opt, sched, batch_idx):
        if self.hparams.debug:
            torch.set_anomaly_enabled(True)

        opt.zero_grad(True)
        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)

        k = enc.A(y) + noise
        Ahk = enc.AH(k)

        if self.net.y0iterations > 0:
            with torch.no_grad():
                ynoisy = cg(
                    enc.AHA, Ahk, Ahk, maxiter=self.net.y0iterations, dims=(-1, -2, -3)
                )
        else:
            ynoisy = Ahk

        rel_iteration = (
            torch.rand(x.shape[0], device=y.device)[:, None, None, None] ** 2
        )
        y_iter = rel_iteration * y + (1 - rel_iteration) * ynoisy

        loss = 0
        if self.net.net:
            if self.hparams.iterateNet:
                xp = self.net._apply_xnet(y_iter, rel_iteration)
            else:
                xp = self.net._apply_xnet(ynoisy, 0 * rel_iteration)
            lossx = (
                nn.functional.mse_loss(x, xp, reduction="none")
                * (0.95 * mask.float() + 0.05)
            ).mean()
            loss = loss + lossx
        if self.net.ynet:
            if self.hparams.yiterateNet:
                yp = self.net._apply_ynet(y_iter, rel_iteration)
            else:
                yp = self.net._apply_ynet(ynoisy, 0 * rel_iteration)
            lossy = (
                nn.functional.mse_loss(
                    torch.view_as_real(yp), torch.view_as_real(y), reduction="none"
                )
                * (0.9 * mask[..., None].float() + 0.1)
            ).mean()
            loss = loss + lossy

        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), 10.0, error_if_nonfinite=True
        )
        opt.step()
        sched.step()

        if self.hparams.debug:

            def log_image(images, name, step, **kwargs):
                import matplotlib.pyplot as plt
                import numpy as np

                images = images.detach().cpu().numpy()
                fig, axs = plt.subplots(
                    images.shape[1],
                    images.shape[0],
                    tight_layout=True,
                    figsize=(images.shape[0] * 2.5, images.shape[1] * 2),
                    dpi=72,
                )

                def plot(ax, data, cutpct=1, sym=False, **kwargs):
                    vmin, vmax = (
                        np.nanpercentile(data, cutpct),
                        np.nanpercentile(data, 100 - cutpct),
                    )
                    if sym:
                        vmax = max(np.abs(vmax), np.abs(vmin))
                        vmin = min(-np.abs(vmin), -np.abs(vmax))
                        kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
                    c = ax.matshow(data, vmin=vmin, vmax=vmax, **kwargs)
                    plt.colorbar(c, ax=ax)

                for axs, ims in zip(axs.T, images):
                    for ax, im in zip(axs, ims):
                        plot(ax, im, cmap="gray", cutpct=0.0)

                self.run[f"debug/image/{name}"].log(fig, step)
                plt.close()

            step = (
                self.trainer.current_epoch
                + batch_idx / self.trainer.num_training_batches
            )
            if batch_idx % 4 == 0:
                if self.net.ynet is not None:
                    log_image(yin[:4].abs(), "y_in", step)
                    log_image(yp[:4].abs(), "y_p", step)
            self.run["debug/loss"].log(loss.item(), step=step)
            for c in range(xp.shape[1]):
                v = xp[:, c].detach()
                l = (
                    torch.nn.functional.mse_loss(x[:, c], v, reduction="none")
                    * mask[:, 0]
                ).sum() / mask[:, 0].sum()
                self.run[f"debug/step0/channel{c}/mse"].log(l.item(), step=step)
            self.run["debug/lossx"].log(lossx.item(), step=step)
            if self.net.ynet:
                self.run["debug/lossy"].log(lossy.item(), step=step)
                lossyin = (
                    nn.functional.mse_loss(
                        torch.view_as_real(yin), torch.view_as_real(y), reduction="none"
                    )
                    * (0.9 * mask[..., None].float() + 0.1)
                ).mean()
                self.run["debug/lossyin"].log(lossyin.item(), step=step)
            for i, p in enumerate(opt.param_groups):
                self.run[f"debug/lr{i}"].log(p["lr"], step)
            if self.net.iterationEmbedding is not None and batch_idx % 4 == 0:
                test = self.net.iterationEmbedding(
                    torch.linspace(0, 1, 4, device=yin.device)
                )
                self.run["debug/emb/min"].log(test.amin(), step)
                self.run["debug/emb/max"].log(test.max(), step)
                self.run["debug/emb/mean"].log(test.mean(), step)
                self.run["debug/emb/std"].log(test.std(), step)

        return loss

    def train_step(self, batch, opt, sched, batch_idx):
        # if self.hparams.debug:
        #     torch.set_anomaly_enabled(True)

        opt.zero_grad(True)
        if self.bad > 10:
            raise ValueError("to many bad steps in series!")
        x, mask, noise, *enc_args = batch
        enc = self.getEncodingOperator(*enc_args)
        y = self.q(x)
        k = enc.A(y)
        k += enc.mask(noise)
        p = self(k, enc)
        loss = self.loss(x, p, mask)

        if self.hparams.debug:
            step = (
                self.trainer.current_epoch
                + batch_idx / self.trainer.num_training_batches
            )
            for i, pg in enumerate(opt.param_groups):
                self.run[f"debug/lr{i}"].log(pg["lr"], step)
            self.run["debug/loss"].log(loss.item(), step=step)
            for i, pred in enumerate(p[0]):
                for c in range(pred.shape[1]):
                    v = pred[:, c].detach()
                    l = (
                        torch.nn.functional.mse_loss(x[:, c], v, reduction="none")
                        * mask[:, 0]
                    ).sum() / mask[:, 0].sum()
                    self.run[f"debug/step{i}/channel{c}/mse"].log(l.item(), step=step)
                    self.run[f"debug/step{i}/channel{c}/min"].log(
                        v.amin().item(), step=step
                    )
                    self.run[f"debug/step{i}/channel{c}/max"].log(
                        v.amax().item(), step=step
                    )
                    self.run[f"debug/step{i}/channel{c}/mean"].log(
                        v.mean().item(), step=step
                    )
            for i, pred in enumerate(p[2]):
                for c in range(pred.shape[1]):
                    v = pred[:, c].detach()
                    l = (
                        torch.nn.functional.mse_loss(x[:, c], v, reduction="none")
                        * mask[:, 0]
                    ).sum() / mask[:, 0].sum()
                    self.run[f"debug/step{i}/xreg_channel{c}/mse"].log(
                        l.item(), step=step
                    )
                    self.run[f"debug/step{i}/xreg_channel{c}/min"].log(
                        v.amin().item(), step=step
                    )
                    self.run[f"debug/step{i}/xreg_channel{c}/max"].log(
                        v.amax().item(), step=step
                    )
                    self.run[f"debug/step{i}/xreg_channel{c}/mean"].log(
                        v.mean().item(), step=step
                    )
            for c in range(x.shape[1]):
                v = x[:, c].detach()
                self.run[f"debug/gt/channel{c}/min"].log(v.amin().item(), step=step)
                self.run[f"debug/gt/channel{c}/max"].log(v.amax().item(), step=step)
                self.run[f"debug/gt/channel{c}/mean"].log(v.mean().item(), step=step)
            for name, weight in self.net.lambdas.named_parameters():
                self.run[f"debug/lambda_softplus_{name}"].log(
                    self.net.softplus(weight.data).detach().cpu(), step=step
                )
            if self.net.iterationEmbedding is not None:
                test = self.net.iterationEmbedding(
                    torch.linspace(0, 1, 4, device=x.device)
                )
                self.run["debug/emb/min"].log(test.amin().item(), step)
                self.run["debug/emb/max"].log(test.max().item(), step)
                self.run["debug/emb/mean"].log(test.mean().item(), step)
                self.run["debug/emb/std"].log(test.std().item(), step)
            if self.net.ynet is not None:
                for i, pred in enumerate(p[4]):
                    v = torch.view_as_real(pred.detach())
                    m = mask[..., None].expand_as(v)
                    l = (
                        torch.nn.functional.mse_loss(
                            torch.view_as_real(y), v, reduction="none"
                        )
                        * m
                    ).sum() / m.sum()
                    self.run[f"debug/step{i}/yreg2/mse"].log(l.item(), step=step)

        if loss > 5 and self.trainer.global_step > 2:
            print("Loss too high, no backprop done", loss)
            sched.step()
            self.bad += 1
            return loss

        self.manual_backward(loss)

        try:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), 1.0, error_if_nonfinite=True
            )
        except Exception as e:
            self.bad += 1
            print("grad error:", e)
            for n, p in self.named_parameters():
                if p.grad is None:
                    continue
                norm = torch.norm(p.grad)
                if (not torch.all(torch.isfinite(p.grad))) or torch.logical_or(
                    norm.isnan(), norm.isinf()
                ):
                    print(n, p.grad.ravel()[:5])
                    p.grad.fill_(0.0)
            for p in self.parameters():
                if p.grad is None:
                    continue
                norm = torch.norm(p.grad)
                if (not torch.all(torch.isfinite(p.grad))) or torch.logical_or(
                    norm.isnan(), norm.isinf()
                ):
                    p.grad.fill_(0.0)
            opt.step()
            sched.step()
            return loss
        opt.step()
        sched.step()
        self.bad = 0
        return loss

    def configure_optimizers(self):
        p_default, p_lambdas, p_ynet, p_emb = [], [], [], []
        for name, p in self.net.named_parameters():
            if "lambdas" in name:
                p_lambdas.append(p)
            elif "ynet" in name:
                p_ynet.append(p)
            elif "iterationEmbedding" in name:
                p_emb.append(p)
            else:
                p_default.append(p)

        optim = torch.optim.AdamW(
            [
                {
                    "params": p_default + p_ynet,
                    "lr": self.hparams.lr_net,
                    "weight_decay": self.hparams.weight_decay,
                    "betas": (0.9, 0.999),
                },
                {
                    "params": p_lambdas,
                    "lr": self.hparams.lr_lam,
                    "weight_decay": self.hparams.weight_decay * 1e-3,
                    "betas": (0.9, 0.999),
                },
                {
                    "params": p_emb,
                    "lr": self.hparams.lr_net / 2,
                    "weight_decay": 1e-2,
                    "betas": (0.9, 0.999),
                },
            ]
        )
        optimPre = torch.optim.AdamW(
            [
                {"params": p_default, "lr": 3e-4, "weight_decay": 1e-2},
                {"params": p_ynet, "lr": 3e-3, "weight_decay": 1e-6},
                {"params": p_emb, "lr": 1e-4, "weight_decay": 1e-4},
            ]
        )

        stepsTotal = self.trainer.fit_loop.max_epochs
        self.trainer.fit_loop.max_epochs = stepsTotal - self.hparams.pretrain_length
        batches = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = self.hparams.pretrain_length
        batchesPre = self.trainer.estimated_stepping_batches
        self.trainer.fit_loop.max_epochs = stepsTotal

        sched = WarmupLR(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                batches - self.hparams.warmup_length,
                eta_min=self.hparams.min_lr,
                verbose=False,
            ),
            self.hparams.min_lr,
            self.hparams.warmup_length,
        )

        warmupPre = max(int(self.hparams.warmup_length / batches * batchesPre), 100)
        schedPre = WarmupLR(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimPre,
                batchesPre - warmupPre,
                eta_min=min(1e-5, self.hparams.min_lr),
                verbose=False,
            ),
            min(1e-5, self.hparams.min_lr),
            warmupPre,
        )
        return [optim, optimPre], [sched, schedPre]
