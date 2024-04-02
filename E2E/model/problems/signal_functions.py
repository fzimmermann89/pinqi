import torch
from typing import Optional, Tuple
from torch import nn, Tensor
import abc


class SignalFunction(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self.zero = torch.zeros(1)

    @property
    @abc.abstractmethod
    def nIn(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def nOut(self) -> int:
        pass

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        return x

    @property
    def bounds(self) -> Tensor:
        if hasattr(self, "_bounds"):
            return self._bounds[None, :, None, None, :]
        else:
            return self.zero[None, None, None, None].expand(1, self.nOut, 1, 1).clone()

    @property
    def initial_values(self) -> Tensor:
        if hasattr(self, "_initial_values"):
            return self._initial_values[None, :, None, None]
        elif hasattr(self, "_bounds"):
            self._bounds[None, :, None, None, :].mean(-1)
        else:
            return

    @property
    def scale(self) -> Tensor:
        if hasattr(self, "_scale"):
            return self._scale[None, :, None, None]
        elif hasattr(self, "_bounds"):
            self._bounds[None, :, None, None, :].std(-1)
        else:
            return


class T1Inversion(SignalFunction):
    def __init__(self, ti: Tuple = (0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
        ti = torch.tensor(ti)[None, :, None, None]
        super().__init__()
        self.register_buffer("ti", ti)
        self.register_buffer("_bounds", torch.tensor(((-3.0, 4.0), (-3.0, 3.0), (-2.0, 10.0))))
        self.register_buffer("_initial_values", torch.tensor((0.9, 0.0, 0.8)))
        self.register_buffer("_scale", torch.tensor((0.2, 0.1, 0.6)))

    @property
    def nOut(self):
        return int(self.ti.numel())

    @property
    def nIn(self):
        return 3

    def forward(self, x: Tensor):
        rpd, ipd, r1 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        pd = rpd + 1j * ipd
        signal = pd * (1 - 2 * torch.exp(-self.ti * r1))
        return signal


class T1Saturation(SignalFunction):
    def __init__(self, ti: Tuple = (0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)):
        ti = torch.tensor(ti)[None, :, None, None]
        super().__init__()
        self.register_buffer("ti", ti)
        # self.register_buffer("_bounds", torch.tensor(((-3.0, 4.0), (-3.0, 3.0), (-2.0, 10.0))))
        self.register_buffer("_bounds", torch.tensor(((-2.0, 3.0), (-2.0, 2.0), (-1.0, 10.0))))

        self.register_buffer("_initial_values", torch.tensor((0.9, 0.0, 0.8)))
        self.register_buffer("_scale", torch.tensor((0.2, 0.1, 0.6)))

    @property
    def nOut(self):
        return int(self.ti.numel())

    @property
    def nIn(self):
        return 3

    def forward(self, x: Tensor):
        rpd, ipd, r1 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        pd = rpd + 1j * ipd
        signal = pd * (1 - torch.exp(-self.ti * r1))
        return signal
