import torch
from torch import nn, Tensor
from typing import Tuple
from functools import partial


class mantisUnet(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, factors: Tuple[int, ...] = (2, 2, 2, 2, 1, 1), filters: int = 64):
        super().__init__()
        conv = partial(nn.Conv2d, kernel_size=4, stride=2, padding=1)
        convT = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1)

        self.enc = nn.ModuleList([nn.Sequential(conv(channels_in, filters, bias=False), nn.BatchNorm2d(filters))])
        encblock = lambda filters, factor: nn.Sequential(nn.ReLU(inplace=True), conv(filters, filters * factor, bias=False), nn.BatchNorm2d(filters * factor))
        encfilters = [filters]
        for f in factors[1:]:
            self.enc.append(encblock(encfilters[-1], f))
            encfilters.append(encfilters[-1] * f)

        decblock = lambda filters_in, filters_out: nn.Sequential(nn.ReLU(inplace=True), convT(filters_in, filters_out, bias=False), nn.BatchNorm2d(filters_out))
        flast = 0
        self.dec = nn.ModuleList()
        for fout, fsc in zip(encfilters[-2::-1], encfilters[-1::-1]):
            fin = flast + fsc
            flast = fout
            self.dec.append(decblock(fin, fout))
        self.dec.append(nn.Sequential(nn.ReLU(inplace=True), convT(2 * filters, channels_out, bias=True)))

    def forward(self, x: Tensor) -> Tensor:
        z = []
        for m in self.enc:
            x = m(x)
            z.append(x)
        x = z.pop()
        for m in self.dec[:-1]:
            x = torch.cat([z.pop(), m(x)], 1)
        x = self.dec[-1](x)
        return x
