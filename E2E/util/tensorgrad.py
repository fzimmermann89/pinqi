from functools import partial
import torch


class GradOperators(torch.nn.Module):
    @staticmethod
    def diff_kernel(ndim, mode):
        if mode == "doublecentral":
            kern = torch.tensor((-1, 0, 1))
        if mode == "central":
            kern = torch.tensor((-1, 0, 1)) / 2
        elif mode == "forward":
            kern = torch.tensor((0, -1, 1))
        elif mode == "backward":
            kern = torch.tensor((-1, 1, 0))
        else:
            raise ValueError(
                f"mode should be one of (central, forward, backward, doublecentral), not {mode}"
            )
        kernel = torch.zeros(ndim, 1, *(ndim * (3,)))
        for i in range(ndim):
            idx = tuple([i, 0, *(i * (1,)), slice(None), *((ndim - i - 1) * (1,))])
            kernel[idx] = kern
        return kernel

    def __init__(self, dim=2, mode="doublecentral", padmode="circular"):
        super().__init__()
        self.register_buffer("kernel", self.diff_kernel(dim, mode))
        self._dim = dim
        self._conv = (
            torch.nn.functional.conv1d,
            torch.nn.functional.conv2d,
            torch.nn.functional.conv3d,
        )[dim - 1]
        self._convT = (
            torch.nn.functional.conv_transpose1d,
            torch.nn.functional.conv_transpose2d,
            torch.nn.functional.conv_transpose3d,
        )[dim - 1]
        self._pad = partial(torch.nn.functional.pad, pad=2 * dim * (1,), mode=padmode)
        if mode == "central":
            self._norm = (self.dim) ** (1 / 2)
        else:
            self._norm = (self.dim * 4) ** (1 / 2)

    @property
    def dim(self):
        return self._dim

    def apply_G(self, x):
        if x.is_complex():
            xr = torch.view_as_real(x).moveaxis(-1, 0)
        else:
            xr = x
        xr = xr.reshape(-1, 1, *x.shape[-self.dim :])
        xp = self._pad(xr)
        y = self._conv(xp, weight=self.kernel, bias=None, padding=0)
        if x.is_complex():
            y = y.reshape(2, *x.shape[: -self.dim], self.dim, *x.shape[-self.dim :])
            y = torch.view_as_complex(y.moveaxis(0, -1).contiguous())
        else:
            y = y.reshape(*x.shape[0 : -self.dim], self.dim, *x.shape[-self.dim :])
        return y

    def apply_GH(self, x):
        if x.is_complex():
            xr = torch.view_as_real(x).moveaxis(-1, 0)
        else:
            xr = x
        xr = xr.reshape(-1, self.dim, *x.shape[-self.dim :])
        xp = self._pad(xr)
        y = self._convT(xp, weight=self.kernel, bias=None, padding=2)
        if x.is_complex():
            y = y.reshape(2, *x.shape[: -self.dim - 1], *x.shape[-self.dim :])
            y = torch.view_as_complex(y.moveaxis(0, -1).contiguous())
        else:
            y = y.reshape(*x.shape[: -self.dim - 1], *x.shape[-self.dim :])
        return y

    def apply_GHG(self, x):
        if x.is_complex():
            xr = torch.view_as_real(x).moveaxis(-1, 0)
        else:
            xr = x
        xr = xr.reshape(-1, 1, *x.shape[-self.dim :])
        xp = self._pad(xr)
        tmp = self._conv(xp, weight=self.kernel, bias=None, padding=0)
        tmp = self._pad(tmp)
        y = self._convT(tmp, weight=self.kernel, bias=None, padding=2)
        if x.is_complex():
            y = y.reshape(2, *x.shape)
            y = torch.view_as_complex(y.moveaxis(0, -1).contiguous())
        else:
            y = y.reshape(*x.shape)
        return y

    def forward(self, x, direction=1):
        if direction > 0:
            return self.apply_G(x)
        elif direction < 0:
            return self.apply_GH(x)
        else:
            return self.apply_GHG(x)

    @property
    def normGHG(self):
        return self._norm


def SmoothTV(x, gradOp, mu=1e-3):
    g = gradOp.apply_G(x)
    gmag = torch.norm(g, p=2, dim=(-gradOp.dim - 1), keepdim=True)
    tv = torch.where(gmag < mu, (gmag.square() / (2 * mu) + mu / 2), gmag)
    return tv.sum()


if __name__ == "__main__":
    v = torch.randn(3, 4, 5, 5) + 1j * torch.randn(3, 4, 5, 5)
    Gop = GradOperators(2)
    Gv = Gop.apply_G(v)
    w = torch.randn_like(Gv)
    GHw = Gop.apply_GH(w)

    def sp(a, b):
        r = (torch.view_as_real(a) * torch.view_as_real(b)).sum(
            tuple(range(1, torch.view_as_real(a).ndim))
        )
        return r

    torch.testing.assert_close(sp(GHw, v), sp(w, Gv), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(Gop(Gv, -1), Gop.apply_GHG(v), rtol=1e-6, atol=1e-6)
