import torch
from torch import tensor, Tensor
from typing import Union, Tuple
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd

from ..util import random_poly2d, trunc_norm


def randmult(input: Tensor, sigma: Union[Tensor, float]) -> Tensor:
    return input * trunc_norm(1, sigma, 1 - 2 * sigma, 1 + 2 * sigma, size=(len(input), 1, 1))


def correlatednoise(img: Tensor, strength: float = 0.1, scale: Tuple[int] = (24, 64), mode: str = "add"):
    if img.ndim <= 2:
        img = img[None, ...]
    correlation_scales = (torch.randint(*scale, (2,))).numpy()
    noise = np.asarray(torch.rand(*img.shape) - 0.5)
    noise = snd.gaussian_filter(noise, (0, *correlation_scales), truncate=3)
    noise = (noise - np.mean(noise)) / np.std(noise)
    s = np.array(strength).reshape(-1, 1, 1)
    if mode == "add":
        img = torch.as_tensor(img) + (s * noise).astype(np.float32)
    elif mode == "mul":
        img = torch.as_tensor(img) * (1 + s * noise).astype(np.float32)
    elif mode == "invadd":
        img = 1 / ((1 / torch.as_tensor(img)) + (s * noise).astype(np.float32))
    else:
        raise NotImplemented
    return torch.squeeze(img)


def randomadd(img: Tensor, strength: float = 0.15) -> Tensor:
    nanmeans = tensor([torch.nansum(i) / i.nelement() for i in img])
    offset = torch.randn(1) * strength * nanmeans
    return torch.abs((offset * nanmeans)[:, None, None] + img)


def polynoise(img: Tensor, strength: float, mode: str = "add") -> Tensor:
    s = np.atleast_1d(strength)[:, None, None]
    noise = np.stack([random_poly2d(*img.shape[-2:], 1, 0.75) for i in range(len(s))])
    if mode == "add":
        img = torch.as_tensor(img) + (s * noise).astype(np.float32)
    elif mode == "mul":
        img = torch.as_tensor(img) * (1 + s * noise).astype(np.float32)
    elif mode == "invadd":
        img = 1 / ((1 / torch.as_tensor(img)) + (s * noise).astype(np.float32))
    else:
        raise NotImplemented
    return np.abs(img)


class RandomPhase:
    def __init__(self, p: float = 1, magnitude: bool = True, strength: float = 1.0, scale: Tuple[int] = (16, 32)):
        self.magnitude = magnitude
        self.strength = strength
        self.p = p
        self.scale = scale

    def __call__(self, img: Tensor) -> Tensor:
        if torch.rand(1) > self.p:
            return img
        mask = torch.isfinite(img)
        img[~mask] = img[mask].mean()

        correlation_scale = torch.randint(*self.scale, (1,)).item()
        kernel = np.outer(*(ss.windows.gaussian(8 * correlation_scale, (correlation_scale)),) * 2)
        noise = np.array(torch.randn(*img.shape[-2:]))
        noise = ss.fftconvolve(noise, kernel, mode="same")
        r = tensor((noise - np.mean(noise)) / np.std(noise) * (2 * np.pi * self.strength))
        p = torch.cos(r) + 1j * torch.sin(r)
        img = torch.fft.ifft2(torch.fft.fft2(img) * p)
        if self.magnitude:
            img = torch.abs(img)
        return img
