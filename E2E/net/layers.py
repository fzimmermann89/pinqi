from typing import Union, Optional, Tuple, Callable, Any
from functools import partial
import torch
from torch import nn, Tensor
import numpy as np
import math
from einops import rearrange
import einops.layers.torch
from abc import abstractmethod, ABC


class StochasticDepth(nn.Module):
    def __init__(self, p: float):
        """
        Deep Networks with Stochastic Depth
            https://arxiv.org/abs/1603.09382
        """
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        survival_rate = 1.0 - self.p
        noise = torch.empty([input.shape[0]] + [1] * (input.ndim - 1), dtype=input.dtype, device=input.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return input * noise

    def __repr__(self) -> str:
        return f"StochasticDepth(p={self.p})"


class JoinMixin(ABC, nn.Module):
    def __init__(self, on_mismatch="pad"):
        super().__init__()
        self.on_mismatch = on_mismatch

        modes = ["fail", "crop", "pad"]
        if not any([mode in on_mismatch for mode in modes]):
            raise ValueError(f"unknow on_mismatch mode: {on_mismatch}. known are {modes}.")

        padmodes = ["reflect", "replicate", "circular"]
        for padmode in padmodes:
            if padmode in on_mismatch:
                self.padmode = padmode
                break
        else:
            self.padmode = "constant"

    def fix_shape(self, x):
        if self.on_mismatch == "fail":
            return x
        elif self.on_mismatch == "crop":
            minshape = (np.inf,) * (x[0].ndim - 2)
            for c in x:
                minshape = tuple(min(m, s) for m, s in zip(minshape, c.shape[2:]))
            xn = []
            for c in x:
                if c.shape[2:] > minshape:
                    newshape = (slice(None), slice(None)) + tuple((slice((s - m) // 2, (s - m) // 2 + m) for s, m in zip(c.shape[2:], minshape)))
                    xn.append(c[newshape])
                else:
                    xn.append(c)
        elif "pad" in self.on_mismatch:
            maxshape: tuple[int, ...] = (0,) * (x[0].ndim - 2)
            for c in x:
                maxshape: tuple[int, ...] = tuple(int(max(m, s)) for m, s in zip(maxshape, c.shape[2:]))
            xn = []
            for c in x:
                if c.shape[2:] < maxshape:
                    pad = tuple(reversed([p for s, m in zip(c.shape[2:], maxshape) for p in (0, int(m - s))]))
                    n = torch.nn.functional.pad(c, pad, mode=self.padmode)
                    xn.append(n)
                else:
                    xn.append(c)
        else:
            xn = x
        return xn


class Concat(JoinMixin):
    def forward(self, x):
        x = self.fix_shape(x)
        return torch.cat(x, 1)


class Add(JoinMixin):
    def forward(self, x):
        x = self.fix_shape(x)
        return sum(x)


def ConvNd(dim: int):
    return [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]


def ConvTransposeNd(dim: int):
    return [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dim - 1]


class WSConv(nn.modules.conv._ConvNd, ABC):
    """https://arxiv.org/pdf/1903.10520v2.pdf"""

    f: Callable[..., Tensor]

    def forward(self, x) -> Tensor:
        if x.dtype == torch.float32:
            eps = 1e-7
        elif x.dtype == torch.float64:
            eps = 1e-5
        else:
            eps = 1e-3
        weight: Tensor = self.weight
        mean = weight.mean(dim=tuple(range(1, weight.ndim)), keepdim=True)
        var = weight.var(tuple(range(1, weight.ndim)), keepdim=True, unbiased=False)
        weight = (weight - mean) * (var + eps).rsqrt()
        return self.f(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSConv1d(WSConv, nn.Conv1d):
    """Weight Standardized 1d Convolution"""

    f = nn.functional.conv1d


class WSConv2d(WSConv, nn.Conv2d):
    """Weight Standardized 2d Convolution"""

    f = nn.functional.conv2d


class WSConv3d(WSConv, nn.Conv3d):
    """Weight Standardized 3d Convolution"""

    f = nn.functional.conv3d


class WSConvTranspose1d(WSConv, nn.ConvTranspose1d):
    """Weight Standardized Transposed 1d Convolution"""

    f = nn.functional.conv_transpose1d


class WSConvTranspose2d(WSConv, nn.ConvTranspose2d):
    """Weight Standardized Transposed 2d Convolution"""

    f = nn.functional.conv_transpose2d


class WSConvTranspose3d(WSConv, nn.ConvTranspose3d):
    """Weight Standardized Transposed 3d Convolution"""

    f = nn.functional.conv_transpose3d


def WSConvNd(dim: int):
    return (WSConv1d, WSConv2d, WSConv3d)[dim - 1]


def WSConvTansposeNd(dim: int):
    return (WSConvTranspose1d, WSConvTranspose2d, WSConvTranspose3d)[dim - 1]


def NormNd(
    normtype: str,
    dim: int,
):
    if normtype == "batch":
        return [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dim - 1]
    if normtype == "instance":
        return partial([nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][dim - 1], affine=True)
    if normtype == "layer":
        return nn.LayerNorm
    if "group" in normtype:
        gstr = normtype.split("group")[1]
        g = int(gstr) if gstr.isdigit() else 8
        return lambda num_channels, *args, **kwargs: nn.GroupNorm(num_groups=g, num_channels=num_channels, *args, **kwargs)

    raise ValueError("normtype must be batch, instance or layer")


def DropoutNd(dim: int):
    return [nn.Dropout, nn.Dropout2d, nn.Dropout3d][dim - 1]


def MaxPoolNd(dim: int):
    return [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dim - 1]


def AvgPoolNd(dim: int):
    return [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d][dim - 1]


def AdaptiveAvgPoolnD(dim):
    return (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)[dim - 1]


class Residual(nn.Module):
    def __init__(self, block, residual=nn.Identity(), after=nn.Identity()):
        super().__init__()
        self.block = block
        self.residual = residual
        self.after = after

    def forward(self, x):
        return self.after(self.block(x) + self.residual(x))


class Sequence(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.steps = nn.ModuleList(modules)

    def forward(self, x):
        ret = [x]
        for m in self.steps:
            x = m(x)
            ret.append(x)
        return ret


class Upsample(nn.Module):
    def __init__(self, dim: int, factor: Union[float, Tuple[float, ...]] = 2, mode: str = "linear", conv_channels: Optional[Tuple[int, int]] = None, keep_leading_dim=False):
        super().__init__()

        rdim = dim - keep_leading_dim
        if mode in ("nearest", "linear", "cubic"):
            if mode == "linear" and 1 <= rdim <= 3:
                mode = ("linear", "bilinear", "trilinear")[rdim - 1]
            elif dim == 2 and mode == "cubic":
                mode = "bicubic"
            elif mode == "nearest":
                mode = "nearest"
            else:
                raise ValueError(f"{mode=} not possible for {rdim=}")
        else:
            raise NotImplementedError(f"unknown mode {mode}")

        sample = nn.Upsample(scale_factor=factor, mode=mode, align_corners=None if mode == "nearest" else False)

        if keep_leading_dim:
            sample = EinopsToAndFrom("b c h ...", "b (c h) ...", sample)

        if conv_channels is not None:
            in_channels, out_channels = conv_channels
            conv = ConvNd(dim)(in_channels, out_channels, kernel_size=(1,) * keep_leading_dim + rdim * (3,), bias=False, padding="same", padding_mode="replicate")
            self.op = nn.Sequential(sample, conv)
        else:
            self.op = sample

    def forward(self, x):
        return self.op(x)


class DownShuffle(nn.Module):
    def __init__(self, dim: int, filter_in: int, filter_out: Optional[int] = None, factor: Union[int, Tuple[int, ...]] = 2):
        """
        ND-Version of PixelShuffle based Downsampling (Reshaping + Conv)
        dim: dimension
        filter_in: input filter size
        filter_out: output filter size, default: input filter size
        factor: scaling factor (int) or tuple of length dim
        """
        super().__init__()
        if isinstance(factor, int):
            factor = (factor,) * dim
        if filter_out is None:
            filter_out = filter_in

        if dim == 1:
            shuffle = einops.layers.torch.Rearrange("b c (d f0) -> b (c f0) d", f0=factor[0])
            conv = nn.Conv1d(filter_in * factor[0], filter_out, 1, bias=False)
        elif dim == 2:
            shuffle = einops.layers.torch.Rearrange("b c (d f0) (h f1)  -> b (c f0 f1) d h", f0=factor[0], f1=factor[1])
            conv = nn.Conv2d(filter_in * factor[0] * factor[1], filter_out, 1, bias=False)
        elif dim == 3:
            shuffle = einops.layers.torch.Rearrange("b c (d f0) (h f1) (w f2) -> b (c f0 f1 f2) d h w", f0=factor[0], f1=factor[1], f2=factor[2])
            conv = nn.Conv3d(filter_in * factor[0] * factor[1] * factor[2], filter_out, 1, bias=False)
        else:
            raise NotImplementedError(f"not implemented for dimension={dim}")

        self.op = nn.Sequential(
            shuffle,
            conv,
        )

    def forward(self, x):
        return self.op(x)


class UpShuffle(nn.Module):
    def __init__(self, dim: int, filter_in: int, filter_out: Optional[int] = None, factor: Union[int, Tuple[int, ...]] = 2):
        """
        ND-Version of PixelShuffle based Upsamling (Conv + Reshaping)
        dim: dimension
        filter_in: input filter size
        filter_out: output filter size, default: input filter size
        factor: scaling factor (int) or tuple of length dim
        """
        super().__init__()
        if isinstance(factor, int):
            factor = (factor,) * dim
        if filter_out is None:
            filter_out = filter_in

        if dim == 1:
            shuffle = einops.layers.torch.Rearrange("b (c f0) d -> b c (d f0)", f0=factor[0])
            conv = nn.Conv1d(filter_in, filter_out * factor[0], 1, bias=False)
        elif dim == 2:
            shuffle = einops.layers.torch.Rearrange("b (c f0 f1) d h -> b c (d f0) (h f1) ", f0=factor[0], f1=factor[1])
            conv = nn.Conv2d(filter_in, filter_out * factor[0] * factor[1], 1, bias=False)
        elif dim == 3:
            shuffle = einops.layers.torch.Rearrange("b (c f0 f1 f2) d h w -> b c (d f0) (h f1) (w f2)", f0=factor[0], f1=factor[1], f2=factor[2])
            conv = nn.Conv3d(filter_in, filter_out * factor[0] * factor[1] * factor[2], 1, bias=False)
        else:
            raise NotImplementedError(f"not implemented for dimension={dim}")
        self.op = nn.Sequential(conv, shuffle)

    def forward(self, x):
        return self.op(x)


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp, in the backward pass like the identity function (gradient 1).
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/3
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    input (Tensor): the input tensor.
    min (Number or Tensor, optional): lower-bound of the range to be clamped to
    max (Number or Tensor, optional): upper-bound of the range to be clamped to
    """
    return DifferentiableClamp.apply(input, min, max)


class ScaledTanh(nn.Module):
    def __init__(self, beta=5):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return torch.tanh(1 / self.beta * x) * self.beta


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=32, n=100.0):
        super().__init__()
        f = -math.log(n) / (dim // 2 - 1)
        w = torch.exp(torch.arange(dim // 2) * f)
        self.register_buffer("w", w)
        self.n = n

    def forward(self, x):
        emb = x[:, None] * self.w
        return torch.cat((emb.sin(), x[:, None] / self.n, emb.cos()[..., :-1]), dim=-1)


class IterationEmbedding(nn.Module):
    def __init__(self, max_n=1.0, dim1=32, dim2=64):
        super().__init__()
        linear = nn.Linear(dim1, dim2)
        nn.init.kaiming_normal_(linear.weight.data)
        self.block = nn.Sequential(
            SinusoidalEmbedding(dim1, max_n * 10),
            linear,
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class AffineConditional(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2),
        )
        self.channels = channels

    def forward(self, x, cond):
        m, b = torch.split(self.mlp(cond), self.channels, -1)
        return m * x + b


class EinopsToAndFrom(nn.Module):
    # https://github.com/lucidrains/einops-exts/blob/main/einops_exts/torch.py, MIT Licence
    # Changed to allow 'changes_' and '...' as axis names not having their size captured
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

        if "..." in from_einops:
            before, after = [part.replace("(", " ").replace(")", " ").strip().split() for part in from_einops.split("...")]
            reconstitute_keys = tuple(zip(before, range(len(before)))) + tuple(zip(after, range(-len(after), 0)))
        else:
            split = from_einops.part.replace("(", " ").replace(")", " ").strip().split()
            reconstitute_keys = tuple(zip(split, range(len(split))))

        self.reconstitute_keys = tuple(filter(lambda x: not "changes_" in x[0], reconstitute_keys))

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = {key: shape[position] for key, position in self.reconstitute_keys}
        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs)
        return x


class ChanLayerNorm(nn.Module):
    """
    Channelwise Layer Norm without Bias
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + self.eps).rsqrt() * self.g


class ScaleLayerNorm(nn.Module):
    """
    LayerNorm without bias
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return nn.functional.layer_norm(x, x.shape[-1:], self.gamma, self.beta)  # type: ignore


class EmbLayer(ABC):
    """
    Any module where forward() take a embedding as second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        ...


class ScaleAndShift(nn.Module,EmbLayer):
    def __init__(self, emb_dim, filters, shift=True):
        super().__init__()
        self.project = nn.Linear(emb_dim, (1 + shift) * filters)
        nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.project.weight)
        self.shift = shift

    def forward(self, x, emb):
        p = self.project(emb)
        p = p[(...,) + ((x.ndim - p.ndim) * (None,))]
        if self.shift:
            scale, shift = torch.chunk(p, 2, 1)
            x = x * (1 + scale)
            x = x + shift
        else:
            x = x * (1 + p)
        return x


def MaybeEmbed(module, x, emb):
    if isinstance(module, EmbLayer):
        return module(x, emb)
    else:
        return module(x)


class ReshapeWrapper(nn.Module):
    def __init__(self, net, complex_to_real=True, real_to_complex=None, channel_to_batch=True):
        super().__init__()
        self.net = net
        self.complex_to_real = complex_to_real
        self.real_to_complex = complex_to_real if real_to_complex is None else real_to_complex
        self.channel_to_batch = channel_to_batch

    def forward(self, x):

        if self.channel_to_batch:
            xin = x.flatten(end_dim=1).unsqueeze(1)
        else:
            xin = x

        if self.complex_to_real:
            if xin.is_complex():
                xin = torch.view_as_real(xin).moveaxis(-1, 1).flatten(start_dim=1, end_dim=2)
            else:
                xin = torch.cat((xin, torch.zeros_like(xin)), 1)

        y = self.net(xin)

        if self.channel_to_batch:
            y = y.reshape(x.shape[0], -1, *y.shape[2:])

        if self.real_to_complex:
            y = y.reshape(y.shape[0], 2, -1, *y.shape[2:])
            y = torch.complex(y[:, 0], y[:, 1])

        return y


def act(name: str = "ReLu"):
    if name.lower() == "relu":
        return partial(nn.ReLU, inplace=True)
    elif name.lower() == "prelu":
        return nn.PReLU
    elif name.lower() == "leakyrelu" or name.lower() == "lrelu":
        return nn.LeakyReLU
    elif name.lower() == "silu":
        return nn.SiLU
    elif name.lower() == "gelu":
        return nn.GELU
    else:
        raise NotImplementedError(name)
