from .layers import *
from .attention import *
from typing import Union, Optional, Tuple, Callable
from functools import partial
import torch
import numpy as np
import math
from einops import rearrange
from itertools import repeat
import collections
from torch import nn


class ResidualCBlock(nn.Module, EmbLayer):
    def __init__(
        self,
        features: Tuple,
        dim: int = 2,
        kernel_size: Union[int, Tuple] = 3,
        dropout: Union[float, nn.Module, None] = None,
        norm: Union[bool, Callable[..., nn.Module], str] = False,
        norm_before_activation=True,
        activation: Optional[Callable[..., nn.Module]] = partial(nn.ReLU, inplace=True),
        bias: bool = True,
        padding: Union[bool, int] = True,
        padding_mode: str = "zeros",
        groups: int = 1,
        final_activation: bool = True,
        final_norm=True,
        emb_dim: int = 0,
        split_dim: bool = False,
        resZero=True,
    ):
        """
        Convolutions from features[0]->features[1]->...->features[-1] with residual connection
        """
        super().__init__()
        if activation is None:
            activation = partial(nn.ReLU, inplace=True)
        self.block = CBlock(features, dim, kernel_size, dropout, norm, norm_before_activation, activation, bias, padding, padding_mode, groups, final_activation=None, split_dim=split_dim, emb_dim=emb_dim, final_norm=final_norm)
        self.resConv = ConvNd(dim)(features[0], features[-1], kernel_size=1, bias=True) if features[0] != features[-1] else None

        if self.block[-1].bias is not None:
            nn.init.zeros_(self.block[-1].bias)

        self.final_activation: Optional[nn.Module] = activation() if final_activation else None
        self.alpha = nn.Parameter(1e-3 * torch.ones(1)) if resZero else None

    def forward(self, x, emb=None):
        ret = self.block(x, emb)
        if self.alpha is not None:
            ret = ret * self.alpha
        if self.resConv:
            x = self.resConv(x)
        ret = ret + x
        if self.final_activation is not None:
            ret = self.final_activation(ret)
        return ret


class SequentialEmb(nn.Sequential, EmbLayer):
    def forward(self, x, emb=None):
        for module in self:
            x = MaybeEmbed(module, x, emb)
        return x


class CBlock(SequentialEmb):
    def __init__(
        self,
        features: Tuple,
        dim: int = 2,
        kernel_size: Union[int, Tuple] = 3,
        dropout: Union[float, nn.Module, None] = None,
        norm: Union[bool, Callable[..., nn.Module], str] = False,
        norm_before_activation=True,
        activation: Optional[Callable[..., nn.Module]] = partial(nn.ReLU, inplace=True),
        bias: bool = True,
        padding: Union[bool, int, str] = True,
        padding_mode: str = "zeros",
        groups: int = 1,
        final_activation=True,
        final_norm=True,
        stride: int = 1,
        emb_dim: int = 0,
        split_dim: bool = False,
    ):
        """
        Convolutions from features[0]->features[1]->...->features[-1] with activation, optional norm and optional dropout
        """

        if padding is True:
            padding = "same"
        if isinstance(dropout, float) and dropout > 0:
            dropout = DropoutNd(dim)(dropout)
        if isinstance(norm, str):
            norm = NormNd(normtype=norm, dim=dim)
        elif norm is True:
            norm = NormNd(normtype="batch", dim=dim)
        if emb_dim > 0:
            embedding = partial(ScaleAndShift, emb_dim=emb_dim, shift=True)

        if isinstance(kernel_size, int):
            conv_kernel_size = split_dim * (1,) + (dim - split_dim) * (kernel_size,)
            split_kernel_size = (kernel_size,) + (dim - 1) * (1,)
        elif split_dim and len(kernel_size) == dim:
            split_kernel_size = (kernel_size[0],) + (dim - 1) * (1,)
            conv_kernel_size = (1,) + tuple(kernel_size[1:])
        else:
            conv_kernel_size = kernel_size

        if split_dim:
            conv_split = partial(ConvNd(dim), kernel_size=split_kernel_size, padding="same", padding_mode="replicate")

        conv = partial(ConvNd(dim), kernel_size=conv_kernel_size, padding=padding, groups=groups, padding_mode=padding_mode, stride=stride)
        modules = []
        for i, (fin, fout) in enumerate(zip(features[:-1], features[1:])):
            modules.append(conv(fin, fout, bias=i == len(features) - 2 if bias == "last" else bias))
            if i == 1 and split_dim:
                modules.append(conv_split(fout, fout, bias=False))
            if dropout:
                modules.append(dropout)
            if norm and norm_before_activation and (final_norm or i < len(features) - 2):
                modules.append(norm(fout))
            if i == 0 and emb_dim > 0:
                modules.append(embedding(filters=fout))
            if activation and (final_activation or i < len(features) - 2):
                modules.append(activation())
            if norm and not norm_before_activation and (final_norm or i < len(features) - 2):
                modules.append(norm(fout))
        if len(features) == 1:
            if norm and norm_before_activation and final_norm:
                modules.append(norm(features[0]))
            if activation and final_activation:
                modules.append(activation())
            if norm and not norm_before_activation and final_norm:
                modules.append(norm(features[0]))
        super().__init__(*modules)

    def __add__(self, other):
        new = type(self)(())
        for m in [*self, *other]:
            new.add_module(str(len(new)), m)
        return new


class RestormerFeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=True)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = nn.functional.silu(x1) * x2
        x = self.project_out(x)
        return x


class RestormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.0):
        super().__init__()
        self.norm1 = ScaleLayerNorm(dim)
        self.attn = TransposedAttention(dim, num_heads)
        self.norm2 = ScaleLayerNorm(dim)
        self.ffn = RestormerFeedForward(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    # from https://github.com/lucidrains/imagen-pytorch/, MIT license
    """
    Transformer block for for applying attention at the end of each layer in a UNet.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        ff_mult: int = 2,
    ):
        """
        dim: Number of channels in the input.
        heads: Number of attention heads
        dim_head: Dimensionality for each attention head
        ff_mult: Channel depth multiplier for the MLP applied after  attention.
        """
        super().__init__()
        self.attn = EinopsToAndFrom("b c h w", "b (h w) c", SelfAttention(dim=dim, heads=heads, dim_head=dim_head))
        self.ff = nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * ff_mult, 1, bias=False),
            nn.SiLU(),
            ChanLayerNorm(dim * ff_mult),
            nn.Conv2d(dim * ff_mult, dim, 1, bias=False),
        )

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


def ntuple(x, n=1):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        dim: int,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ):

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                kernel_size = ntuple(kernel_size, dim)
                dilation = ntuple(dilation, dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(dim))
        if bias is None:
            bias = norm_layer is None
        layers = [ConvNd(dim)(input_channels, output_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)]

        if norm_layer is not None:
            layers.append(norm_layer(output_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.output_channels = output_channels
