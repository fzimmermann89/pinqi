from typing import Union, Optional, Tuple, Callable
from functools import partial
import torch
from torch import nn
from .blocks import *
from .unet import UnetLayer


class MultiResBlock(nn.Module):
    def __init__(
        self,
        features3: Tuple,
        dim: int = 2,
        dropout: Union[float, nn.Module, None] = None,
        norm: Union[bool, Callable[..., nn.Module], str] = False,
        norm_before_activation: bool = True,
        activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        bias: bool = True,
        padding: Union[bool, int] = True,
        padding_mode: str = "zeros",
        groups:int = 1,
    ):
        """
        MultiResBlock
        """
        super().__init__()
        conv3 = partial(ConvNd(dim), kernel_size=3, padding=1 if padding else 0, groups=groups, bias=bias, padding_mode=padding_mode)
        conv1 = partial(ConvNd(dim), kernel_size=1, padding=0, groups=groups, bias=bias, padding_mode=padding_mode)

        modules3 = nn.ModuleList()
        for fin, fout in zip(features3[:-1], features3[1:]):
            block = []
            block.append(conv3(fin, fout))
            if dropout:
                block.append(dropout)
            if norm and norm_before_activation:
                block.append(norm(fout))
            if activation:
                block.append(activation())
            if norm and not norm_before_activation:
                block.append(norm(fout))
            modules3.append(nn.Sequential(*block))
        self.modules3 = modules3

        block = [conv1(features3[0], sum(features3[1:]))]
        if dropout:
            block.append(dropout)
        if norm and norm_before_activation:
            block.append(norm(fout))
        if activation:
            block.append(activation())
        if norm and not norm_before_activation:
            block.append(norm(fout))
        self.modules1 = nn.Sequential(*block)

    def forward(self, x):
        features3 = [x]
        for conv3 in self.modules3:
            features3.append(conv3(features3[-1]))
        features3 = torch.cat(features3[1:], 1)
        features1 = self.modules1(x)
        return features3 + features1


class MultiResSkipBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "zeros",
        activation: Callable = partial(nn.ReLU, inplace=True),
        normtype: str = "batch",
        norm_before_activation: bool = True,
        dropout: Union[float, nn.Module, None] = None,
    ):
        super().__init__()

        blocks = [ConvNd(dim)(in_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if normtype and normtype != "none":
            blocks.append(NormNd(normtype, dim)(out_channels))
        self.res = nn.Sequential(*blocks)

        blocks = [ConvNd(dim)(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)]
        if dropout:
            blocks.append(DropoutNd(dim)(dropout, inplace=True))
        if normtype and normtype != "none" and norm_before_activation:
            blocks.append(NormNd(normtype, dim)(out_channels))
        blocks.append(activation())
        if normtype and normtype != "none" and not norm_before_activation:
            blocks.append(NormNd(normtype, dim)(out_channels))
        self.main = nn.Sequential(*blocks)

        blocks = [activation()]
        if normtype and normtype != "none":
            blocks.append(NormNd(normtype, dim)(out_channels))
        self.final = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.res(x) + self.main(x)
        x = self.final(x)
        return x


class MultiResSkipPath(nn.Module):
    def __init__(self, dim: int, channels_in: int, channels: int, stages: int, padding_mode: str = "zeros", activation: Callable = partial(nn.ReLU, inplace=True), normtype="batch", norm_before_activation: bool = True, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[MultiResSkipBlock(dim, channels_in if i == 0 else channels, channels, padding_mode, activation, normtype, norm_before_activation, dropout) for i in range(stages)])

    def forward(self, x):
        return self.blocks(x)


class MultiResUnet(nn.Module):
    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        layer: int = 3,
        filters: int = 8,
        norm: Union[bool, str, Callable[..., nn.Module]] = False,
        norm_before_activation: bool = True,
        dropout: float = 0.0,
        dropout_last: float = 0.0,
        padding_mode="zeros",
        residual=False,
        up_mode="upconv",
    ):
        """
        MultiResUnet, arXiv:1902.04049

        channels_in: Number of Input channels
        channels_out: Number of Output channels
        layer: Number of layer, counted by down-steps
        filters: Initial number of filters
        norm: False=No; True or 'batch'=BatchNorm; 'instance'=InstanceNorm
        norm_before_activation: Insert norm before activation, otherwise after
        dropout: Dropout in Encoder and Decoder. float=Dropout Propability or nn.Module. 0.=No dropout
        dropout_last: Dropout in final 1-Conv. float=Dropout Propability or nn.Module. 0.=No dropout
        padding_mode: padding mode for Convs
        residual: Use residual on the first min(channels_in,channels_out) channels over whole Net
        up_mode: "upconv"=Transposed Convolution: "upsample": NN-upsampling+Conv
        """

        super().__init__()
        max_pooling_window = 2
        max_pooling_stride = 2

        downsampling = MaxPoolNd(dim)(max_pooling_window, max_pooling_stride)
        if up_mode == "upconv":
            upsampling = partial(ConvTransposeNd(dim), kernel_size=2, stride=2, bias=True)
        elif up_mode == "upsample":
            upsampling = lambda in_channels, out_channels: Upsample(dim=dim, factor=2., up_mode='nearest', conv_channels=(in_channels, out_channels))
        else:
            raise NotImplementedError(f"unknown up_mode {up_mode}")
        encoder = partial(MultiResBlock, dim=dim, norm=norm, bias=True, dropout=dropout, padding=True, padding_mode=padding_mode)

        features_enc = []
        features_dec = []
        features_skip = []
        features_up = []
        for depth in range(0, layer):
            scale = 2 ** (depth)
            features_enc.append([filters * 3 * scale] + [filters * scale, filters * 2 * scale, filters * 3 * scale])
            features_skip.append([sum(features_enc[-1][1:]), 4 * filters * scale])
            features_up.append([None, 4 * filters * scale])
            features_dec.append([features_skip[-1][-1] + features_up[-1][-1], filters * scale, filters * 2 * scale, filters * 3 * scale])
        inner = [2 * f for f in features_enc[-1]]
        for fup, fdec in zip(features_up, features_dec[1:] + [inner]):
            fup[0] = sum(fdec[1:])
        features_enc[0][0] = channels_in

        net = encoder(inner)
        for (
            skiplength,
            fenc,
            fdec,
            fup,
            fskip,
        ) in zip(range(1, layer + 1), features_enc[::-1], features_dec[::-1], features_up[::-1], features_skip[::-1]):
            d = nn.Sequential(Concat(), encoder(fdec))
            e = encoder(fenc)
            u = upsampling(*fup)
            s = MultiResSkipPath(dim, *fskip, skiplength)
            net = UnetLayer(e, downsampling, net, u, d, s)
        self.net = net

        self.last = CBlock((sum(features_dec[0][1:]), channels_out), dropout=dropout_last, bias=True, activation=None)
        self.residualchannels = min(channels_in, channels_out) if residual else False

    def forward(self, x):
        ret = self.net(x)
        ret = self.last(ret)
        if self.residualchannels:
            ret[:, : self.residualchannels, ...] += x[:, : self.residualchannels, ...]
        return ret