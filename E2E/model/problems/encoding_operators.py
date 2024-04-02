import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Optional, Union
from ...util.rnd import random_poly2d


def _uniform(low, high, size=1):
    return ((torch.rand(size) * (high - low)) + low).numpy()


class Cartesian_Operator(torch.nn.Module):
    def __init__(self, csm: Tensor, mask: Tensor):
        super().__init__()
        self.register_buffer("_csm", csm.unsqueeze(-4))
        self.register_buffer("_mask", mask.unsqueeze(-3))

    def _expand(self, x):
        x = x.unsqueeze(-3) * self._csm  # ...,csm,x,y
        return x

    def _reduce(self, x):
        x = (x * self._csm.conj()).sum(-3)  # ...,x,y
        return x

    def _F(self, x):
        x = torch.fft.fft2(x, norm="ortho")
        return x

    def _iF(self, x):
        return torch.fft.ifft2(x, norm="ortho")

    def mask(self, x):
        x = x * self._mask
        return x

    def A(self, x: Tensor) -> Tensor:
        x = self._expand(x)
        x = self._F(x)
        x = self.mask(x)
        return x

    def AH(self, x: Tensor) -> Tensor:
        x = self.mask(x)
        x = self._iF(x)
        x = self._reduce(x)
        return x

    def AHA(self, x: Tensor) -> Tensor:
        x = self._expand(x)
        x = self._F(x)
        x = self.mask(x)
        x = self._iF(x)
        x = self._reduce(x)
        return x

    def EC(self, x: Tensor):
        x = self._expand(x)
        x = self._F(x)
        return x


def cartesian_mask(shape, acceleration, center_n=8, seed=None):
    # based on https://github.com/js3611/Deep-MRI-Reconstruction/blob/master/utils/compressed_sensing.py#L46
    # under http://www.apache.org/licenses/LICENSE-2.0

    def gaussian(N, s):
        return np.exp(-0.5 * (s / 2 * np.linspace(-1, 1, N, endpoint=False) ** 2))

    gen = np.random.default_rng(seed=seed)
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    mask = np.zeros((N, Nx))
    n = int(Nx / acceleration)
    pdf = gaussian(Nx, 10) + 1 / (2.0 * acceleration)

    if center_n:
        pdf[Nx // 2 - center_n // 2 : Nx // 2 - center_n // 2 + center_n] = 0
        mask[:, Nx // 2 - center_n // 2 : Nx // 2 - center_n // 2 + center_n] = 1
        n = max(0, n - center_n)

    pdf /= np.sum(pdf)
    for i in range(N):
        idx = gen.choice(Nx, n, replace=False, shuffle=False, p=pdf)
        mask[i, idx] = 1
    mask = np.broadcast_to(mask[..., None], (N, Nx, Ny)).reshape(shape)
    mask = np.fft.ifftshift(mask, axes=(-1, -2))
    return mask


def generate_csm(
    size: Tuple[int, int],
    Ncoils: int,
    fwhm: Union[None, Tuple[float, float], float] = None,
    phasestrength: Tuple[float, ...] = (1, 0.2),
    seed: Optional[int] = None,
    shuffle: bool = False,
    penalize_edges: bool = True,
    normalized=True,
):
    """
    Generate Gaussian CSMs
    For each coil, the center is choosen randomly with probability 1/coverage of already chosen coils
    size: Size of th CSMs
    Ncoils: Number of maps to gerneate
    fwhm: FWHM of the Gaussians if float, range to choose from if tuple (low,high). default=(3/sqrt(Ncoils),5/sqrt(Ncoils))
    phasestrength: Tuple of floats of the coefficient strength of the phase polynomial, number of values detercmines maximum degree.
    seed: seed value for random generator, if None use default numpy seeding. Function will be deterministic for a given seed.
    """
    if fwhm is None:
        low = 3 / np.sqrt(Ncoils)
        high = 6 / np.sqrt(Ncoils)
    elif isinstance(fwhm, [float, int]):
        low = fwhm
        high = fwhm
    else:
        low, high = fwhm
    coverage = np.ones(size)
    coverage *= 0.01 / coverage.size
    gen = np.random.default_rng(seed)
    csms = []

    for n in range(Ncoils):
        p = 1 / (coverage) ** 2
        if penalize_edges:
            p[: p.shape[0] // 8, :] *= 0.4
            p[-p.shape[0] // 8 :, :] *= 0.4
            p[:, : p.shape[1] // 8] *= 0.4
            p[:, -p.shape[1] // 8 :] *= 0.4
            p[: p.shape[0] // 4, :] *= 0.8
            p[-p.shape[0] // 4 :, :] *= 0.8
            p[:, : p.shape[1] // 4] *= 0.8
            p[:, -p.shape[1] // 4 :] *= 0.8
        p /= p.sum()
        center = (np.array(np.unravel_index(gen.choice(coverage.size, p=p.ravel()), size)) + gen.uniform(-1, 1)) / (np.array(size) / 2) - 1
        w = low if low == high else gen.uniform(low, high)
        x = np.linspace(-1, 1, size[0])[:, None]
        y = np.linspace(-1, 1, size[1])[None, :]
        amp = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / w ** 2)
        amp *= 1 / (np.sqrt(Ncoils * (amp ** 2).mean()))
        polyorder = np.size(phasestrength)
        coeff = gen.uniform(-1 / 2, 1 / 2, (2, polyorder)) * np.asarray(phasestrength)
        phase = (sum(coeff[0, i] * x ** (i + 1) for i in range(polyorder)) + 1) * (sum(coeff[1, i] * y ** (i + 1) for i in range(polyorder)) + 1) - 1
        phase += gen.uniform(0, 2 * np.pi)
        csm = amp * np.exp(1j * phase)
        coverage += amp ** 2
        csms.append(csm)
    if shuffle:
        gen.shuffle(csms)
    csms = np.array(csms)
    if normalized:
        csms /= np.sqrt((csms * csms.conj()).sum(0, keepdims=True))
    return csms


def simple_csm(shape, Ncoils, ny=(2, 4), nx=(4, None), sigma=(40, 60), rand_pos=True):
    if np.size(sigma) == 1:
        sigma = min(shape) * sigma
    elif np.size(sigma) == 2:
        sigma = min(shape) * _uniform(*sigma, size=(Ncoils, 1, 1))
    else:
        raise ValueError("sigma shall be either a number of a tuple of (min,max)")

    def get_n(n):
        if n is None:
            return Ncoils
        elif np.size(n) == 2:
            v = [Ncoils if i is None else min(i, Ncoils) for i in n]
            return torch.randint(v[0], v[1] + 1, (1,))
        else:
            return n

    Nx, Ny = shape
    nx = get_n(nx)
    ny = get_n(ny)

    cx = np.arange(Nx / (nx * 2), Nx, Nx / nx)
    cx = np.tile(cx, int(np.ceil(Ncoils / len(cx))))[:Ncoils]
    cy = np.arange(Ny / (ny * 2), Ny, Ny / ny)
    cy = np.tile(cy, int(np.ceil(Ncoils / len(cy))))[:Ncoils]
    if rand_pos:
        cx = cx + _uniform(-Nx / (nx * 2), Nx / (nx * 2), Ncoils)
        cy = cy + _uniform(-Ny / (ny * 2), Ny / (ny * 2), Ncoils)
        cy = cy[torch.randperm(len(cy))]
    p = 2 * np.pi * torch.rand(Ncoils).numpy()
    tmp = np.exp(-((cx[:, None, None] - np.arange(Nx)[:, None]) ** 2 + (cy[:, None, None] - np.arange(Ny)[None, :]) ** 2) / sigma) * np.exp(1j * p)[:, None, None]
    ret = tmp * (1 / np.sqrt(np.sum(tmp * tmp.conj(), 0)))
    return ret


def circular_csm(imagesize, Ncoils):
    """
    simulate birdcage like sensitivity maps. 
    Based on torchkbnufft example
    Based on a script by Florian Knoll.
    """

    def mrisensesim(size, ncoils=8, coil_width=2, phi=0):
        c_width = coil_width * min(size)
        c_rad = min(size[0:1]) / 2
        smap = []
        yy, xx = np.meshgrid(range(size[1]), range(size[0]), indexing="ij")
        for i in range(ncoils):
            theta = np.radians((i - 1) * 360 / ncoils + phi)
            x0 = c_rad * np.cos(theta) + size[0] / 2
            y0 = c_rad * np.sin(theta) + size[1] / 2
            smap.append(np.exp(-1 * ((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * c_width)))
        return np.array(smap)

    csm = torch.as_tensor(mrisensesim(imagesize, ncoils=Ncoils, coil_width=float(3 + 1 * torch.randn(1).clamp(-2, 2)), phi=float(torch.rand(1) * 360)))
    phase = torch.stack([(0.1 * torch.as_tensor(random_poly2d(*imagesize, (1.0, 0.5, 0.25)))) for i in range(Ncoils)]) + 0.5 * torch.rand(Ncoils, 1, 1)
    phase -= phase[0].mean()
    phase = torch.exp(phase * 2j * np.pi)
    csm = csm * phase
    norm = 1 / ((csm * csm.conj()).sum(0, True).sqrt())
    return (csm * norm).numpy()


def test_adjoint(enc):
    shape = *enc._mask.shape[0:2], *enc._mask.shape[-2:]
    x = (torch.rand(shape) + 1j * torch.rand(shape)).to(device=enc._mask.device)
    Ax = enc.A(x)
    y = torch.rand_like(Ax)
    AHy = enc.AH(y)
    return torch.allclose((y * Ax.conj()).sum(), (AHy * x.conj()).sum())
