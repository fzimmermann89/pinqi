from typing import Union, Optional, Tuple, Callable
from functools import partial
import torch
import numpy as np
import math
from einops import rearrange
from .layers import ConvNd, AdaptiveAvgPoolnD
from torch import nn, Tensor


class TransposedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        """
        Transposed Self Attention from Restormer
        https://github.com/swz30/Restormer
        https://arxiv.org/pdf/2111.09881.pdf
        
        dim: input dimension
        num_heads: number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False,)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = (rearrange(i, "b (head c) h w -> b head c (h w)", head=self.num_heads) for i in qkv.chunk(3, dim=1))
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        sim = (q @ k.transpose(-2, -1)) * self.temperature
        attn = sim.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class SelfAttention(nn.Module):
    """
    Multi-headed self attention
    """

    def __init__(
        self, dim: int, dim_head: int = 64, heads: int = 8,
    ):
        """
        dim: Input dimensionality.
        dim_head: Dimensionality for each attention head.
        heads: Number of attention heads.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.LayerNorm(dim))

    def forward(self, x, mask=None):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale
        sim = torch.einsum("b h i d, b j d -> b h i j", q, k)
        if mask is not None:
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = nn.functional.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SqueezeExcitation(nn.Module):
    """
        Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 
    """

    def __init__(
        self, dim, input_channels: int, squeeze_channels: int, activation: Callable[..., nn.Module] = nn.ReLU, scale_activation: Callable[..., torch.nn.Module] = nn.Sigmoid,
    ):
        super().__init__()
        self.scale = nn.Sequential(AdaptiveAvgPoolnD(dim)(1), ConvNd(dim)(input_channels, squeeze_channels, 1), activation(), ConvNd(dim)(squeeze_channels, input_channels, 1), scale_activation(),)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale(x)
