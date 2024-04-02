from typing import Union, Optional, Tuple, Callable
from functools import partial
import torch
import math
from einops.layers.torch import Rearrange
from torch import nn

from .blocks import *


class UnetLayer(nn.Module, EmbLayer):
    """
    One Layer in a Unet-based Net
    X--> Encoder -------------- Skip--------------- Decoder --> X"
          |-- downsampling -- SubLayers -- upsampling --|

    """

    def __init__(self, encoder, downsampling, sublayer, upsampling, decoder, skip=nn.Identity()):
        super().__init__()
        self.encoder = encoder
        self.downpath = SequentialEmb(downsampling, sublayer, upsampling)
        self.skip = skip
        self.decoder = decoder

    def forward(self, x, emb):
        x = MaybeEmbed(self.encoder, x, emb)
        xdown = MaybeEmbed(self.downpath, x, emb)
        skip = MaybeEmbed(self.skip, x, emb)
        x = MaybeEmbed(self.decoder, (skip, xdown), emb)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        dim: Union[int, float],
        channels_in: int,
        channels_out: int,
        layer: int = 4,
        conv_per_enc_block: int = 2,
        conv_per_dec_block: int = 2,
        filters: int = 32,
        kernel_size: int = 3,
        norm: Union[bool, str, Callable[..., nn.Module]] = False,
        norm_before_activation: bool = True,
        bias: Union[bool, str] = True,
        dropout: float = 0.0,
        dropout_last: float = 0.0,
        padding_mode="zeros",
        residual=False,
        up_mode="linear",
        down_mode="maxpool",
        activation: Union[str, Callable[..., nn.Module]] = "relu",
        feature_growth: Callable[[int], float] = lambda depth: 2.0,
        groups_enc: int = 1,
        groups_dec: int = 1,
        groups_last: int = 1,
        additional_last_decoder: int = 0,
        change_filters_last=True,
        emb_dim=0,
        skip_add=False,
    ):
        """
        A mostly vanilla UNet with linear final activation

        dim: 2 or 3 for 2D and 3D Unet, 2.5 for mostly-2D with an depth-wise conv in each block.
        channels_in: Number of Input channels
        channels_out: Number of Output channels
        layer: Number of layer, counted by down-steps
        conv_per_enc_block: Convolutions per encoder block
        conv_per_dec_block: Convolutions perdecoder block, not counting the upscaling
        filters: Initial number of filters
        kernel_size: size of Kernel in Encoder and Decoder Convolutions
        norm: False=No; True or 'batch'=BatchNorm; 'instance'=InstanceNorm
        norm_before_activation: Insert the norm before the activation, otherwise after the activation
        bias: True=Use bias in all convolutions; last=Use bias only in final 1-Conv
        dropout: Dropout in Encoder and Decoder. float=Dropout Propability or nn.Module. 0.=No dropout
        dropout_last: Dropout in final 1-Conv. float=Dropout Propability or nn.Module. 0.=No dropout
        padding_mode: padding mode for Convs or none.
        residual: Use residual connection between input and output. outer/inner/both=True/none=False
        up_mode: "conv"=Transposed Convolution: "nearest"/"linear"/"cubic": upsamling+Conv. if string contains '_reduce', channels will be reduced.
        down_mode: "conv"=Dilated Convolution, "maxpool", "averagepool" or "downshuffle"
        feature_growth: function depth:int->growth factor of feature number
        groups_enc/_dec/_last: Groups used in all encoder/decoder stages / last 1x1 Conv
        additional_last_decoder: Additional Convs used after last decoder before final output
        change_filters_last: Increase number of filters in last step of a block
        skip_add: Do an add of the skipped connection to the upsampled instead of a concat
        """
        super().__init__()

        if not (isinstance(residual, bool) or residual in ("outer", "inner", "both", "none")):
            raise ValueError("residual must be bool or 'outer'/'inner'/'both'/'none'")

        if isinstance(activation, str):
            activation = act(activation)

        if bias == "last":
            bias_last = True
            bias = False
        else:
            if not isinstance(bias, bool):
                raise ValueError("bias must be bool or 'last'")
            bias_last = bias

        fdim = dim
        dim = int(math.ceil(dim))
        half_dim = math.isclose(fdim - math.floor(fdim), 0.5)

        if down_mode == "maxpool":
            pooling_window = pooling_stride = (1,) * half_dim + (2,) * (dim - half_dim)
            downsampling = lambda *_: MaxPoolNd(dim)(pooling_window, pooling_stride)
        elif down_mode == "averagepool":
            pooling_window = pooling_stride = (1,) * half_dim + (2,) * (dim - half_dim)
            downsampling = lambda *_: AvgPoolNd(dim)(pooling_window, pooling_stride)
        elif down_mode == "conv":
            pooling_window, pooling_stride = (1,) * half_dim + (kernel_size,) * (dim - half_dim), (1,) * half_dim + (2,) * (dim - half_dim)
            downsampling = partial(ConvNd(dim), stride=pooling_stride, kernel_size=pooling_window, padding="same")
        elif down_mode == "downshuffle":
            factor = (1,) * half_dim + (2,) * (dim - half_dim)
            downsampling = partial(DownShuffle(dim=dim, factor=factor))
        else:
            raise NotImplementedError(f"unknown down_mode {down_mode}")

        reduce_upsampling = "reduce" in up_mode

        if "_nofinal" in str(norm):
            norm = str(norm).replace("_nofinal", "")
            final_norm = False
        else:
            final_norm = True

        upm = up_mode.split("_")[0]
        if upm == "conv":
            window = (1,) * half_dim + (kernel_size & ~1,) * (dim - half_dim)
            upsampling = partial(ConvTransposeNd(dim), kernel_size=window, stride=2, bias=bias)
        elif upm in ("nearest", "linear", "cubic"):
            upsampling = lambda in_channels, out_channels: Upsample(dim=dim, factor=2.0, mode=upm, conv_channels=(in_channels, out_channels), keep_leading_dim=half_dim)
        else:
            raise NotImplementedError(f"unknown up_mode {up_mode}")

        block = partial(
            ResidualCBlock if residual in ("inner", True, "both") else CBlock,
            dim=dim,
            kernel_size=kernel_size,
            norm=norm,
            norm_before_activation=norm_before_activation,
            bias=bias,
            dropout=dropout,
            padding=padding_mode != "none",
            activation=activation,
            padding_mode="zeros" if padding_mode == "none" else padding_mode,
            split_dim=half_dim,
            final_norm=final_norm,
        )
        join = Concat("crop" if padding_mode == "none" else "pad")
        features_enc = [(channels_in,) + (filters,) * (conv_per_enc_block - 1) + (int(filters * feature_growth(0)) & ~1,)]
        last = features_enc[-1][-1]
        if skip_add:
            factor = 0
            join = Add()
        elif reduce_upsampling:
            factor = 1
        else:
            factor = feature_growth(1)
        features_dec = [(last + int(factor * last),) + (last,) * conv_per_dec_block]
        for depth in range(1, layer + 1):
            new = int(feature_growth(depth) * last) & ~1
            if change_filters_last:
                features_enc.append((last,) * conv_per_enc_block + (new,))
            else:
                features_enc.append((last,) + conv_per_enc_block * (new,))
            last = features_enc[-1][-1]
            if not (reduce_upsampling or skip_add):
                factor = feature_growth(depth + 1)
            features_dec.append((last + int(factor * last) & ~1,) + (last,) * conv_per_dec_block)
        net = block(features_enc[-1])
        for fenc, fdec, fup, fdown in zip(features_enc[-2::-1], features_dec[-2::-1], features_enc[-1::-1], features_enc[-3::-1] + [[1]]):
            decoder = SequentialEmb(join, block(fdec, groups=groups_dec, emb_dim=emb_dim))
            encoder = block(fenc, groups=groups_enc, emb_dim=emb_dim)
            up = upsampling(fup[-1], fdec[0] if skip_add else fdec[0] - fenc[-1])
            down = downsampling(fenc[-1], fup[0])
            net = UnetLayer(encoder, down, net, up, decoder)
        self.net = net

        self.last = CBlock((features_enc[0][-1], channels_out), dim=dim, kernel_size=1, dropout=dropout_last, bias=bias_last, activation=None, groups=groups_last)
        if additional_last_decoder > 0:
            self.last = block((features_enc[0][-1],) * (additional_last_decoder + 1)) + self.last

        if residual in ("outer", True, "both"):
            self.residual = nn.Identity() if channels_in == channels_out else ConvNd(dim)(channels_in, channels_out, kernel_size=1, bias=False)
        else:
            self.residual = None
        self.emb_dim = emb_dim

    def forward(self, x, emb=None):
        if self.emb_dim > 0 and emb is None:
            emb = torch.zeros(x.shape[0], self.emb_dim, dtype=x.dtype, device=x.device)
        ret = self.net(x, emb)
        ret = self.last(ret)
        if self.residual is not None:
            ret = ret + self.residual(x)
        return ret
