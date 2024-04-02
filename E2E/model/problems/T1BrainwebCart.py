from pathlib import Path
import torch
from functools import partial
import numpy as np
from torchvision import transforms as T
from abc import ABC
import scipy.ndimage as snd


from .base import Problem
from .encoding_operators import Cartesian_Operator, cartesian_mask, simple_csm, circular_csm
from ...data import BrainwebSlices, values_t1t2pd3T
from .signal_functions import T1Inversion, T1Saturation
from ...util import random_poly2d, trunc_norm


def coils_ny(Ncoils):
    if Ncoils % 2 or Ncoils < 4:
        return 1
    elif Ncoils < 16 or Ncoils % 4:
        return 2
    else:
        return 4


class CartUSDataset(torch.utils.data.Dataset):
    def __init__(self, dsx, csm_fun, kmask_fun, Ncoils=8, Nd=8, noise_strength=(0, 0.05), augment_fun=None):
        self.dsx = dsx
        self.csm_fun = csm_fun
        self.kmask_fun = kmask_fun
        self.Ncoils = Ncoils
        self.Nd = Nd
        self.noise_strength = noise_strength
        self.augment_fun = augment_fun

    def __getitem__(self, idx):
        val = self.dsx[idx]
        mask, classes, x = val[0:1].bool(), val[1:2].int(), val[2:].float()
        csm = self.csm_fun(x.shape[-2:], Ncoils=self.Ncoils)
        kmask = self.kmask_fun((self.Nd, *x.shape[-2:]))
        if np.isscalar(self.noise_strength):
            noise_strength = self.noise_strength
        else:
            noise_strength = torch.rand(1) * (self.noise_strength[1] - self.noise_strength[0]) + self.noise_strength[0]
        noise = noise_strength * torch.randn(self.Nd, self.Ncoils, *x.shape[-2:])

        if self.augment_fun:
            x = self.augment_fun(x)

        return (
            x,
            mask,
            classes,
            noise,
            csm.astype(np.complex64),
            kmask.astype(np.float32),
        )

    def __len__(self):
        return len(self.dsx)


class T1BrainwebCartBase(Problem, ABC):
    def __init__(
        self,
        path="data/brainwebClasses/",
        batchsize=8,
        acceleration=8,
        Ncoils=8,
        noise_strength=(0, 0.05),
        cuts=(120, 120, 0, 0, 0, 0),
        center_n=8,
        Npx=192,
        values=values_t1t2pd3T,
        value_augment=False,
    ):
        super().__init__()
        path = Path(path)
        transforms = (
            (
                T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(Npx / 400, Npx / 300), shear=5, fill=0.0, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((Npx, Npx)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ),
            (T.GaussianBlur(5, sigma=(0.01, Npx / 600)),),
        )

        transformsTest = (
            (T.Lambda(lambda x: T.functional.affine(x, angle=180.0, scale=Npx / 390, translate=(0, 0), shear=0.0, fill=0.0, interpolation=T.InterpolationMode.BILINEAR)), T.CenterCrop((Npx, Npx)),),
            (T.GaussianBlur(5, sigma=(0.05)),),
        )

        def augment_values(x):
            outside = x[0] == 0
            p = torch.atan2(x[1], x[0])
            p = p + torch.as_tensor(random_poly2d(*p.shape, (0.1, 0.1, 0.1))).clip(-0.2, 0.2)
            p = (p - p.mean()) + torch.randn(1) * 0.2
            m = (x[0] ** 2 + x[1] ** 2).sqrt()
            noise = snd.gaussian_filter(np.asarray(torch.rand(*m.shape) - 0.5), (torch.rand(2) + 1).numpy() * (min(m.shape) / 12), truncate=2)
            noise = torch.as_tensor((noise - np.mean(noise)) / np.std(noise) * 0.1).clip(-0.1, 0.1)
            m = m * (1 + torch.as_tensor(noise + random_poly2d(*p.shape, (0.3, 0.2, 0.2))).clip(-0.5, 0.3))
            r1 = x[2]
            poly = random_poly2d(*r1.shape, (0.1, 0.2, 0.2))
            ptp = poly.ptp()
            poly = torch.as_tensor((poly - poly.mean()) * (min(ptp, 0.2) / ptp)) * trunc_norm(1.0, 0.1, 0.8, 1.2)
            r1 = r1 + poly * (r1 + 0.3)
            poly = random_poly2d(*r1.shape, (0.1, 0.2, 0.2))
            ptp = poly.ptp()
            poly = torch.as_tensor((poly - poly.mean()) * (min(ptp, 0.2) / ptp)) * trunc_norm(1.0, 0.1, 0.8, 1.2)
            noise = snd.gaussian_filter(np.asarray(torch.rand(*r1.shape) - 0.5), (torch.rand(2) + 1).numpy() * (min(r1.shape) / 12), truncate=2)
            noise = torch.as_tensor((noise - np.mean(noise)) / np.std(noise) * 0.1).clip(-0.1, 0.1)
            r1 = 1 / (1 / r1 + (poly + noise) * (1 / r1 + 0.3))
            r1.clip(min=0.1, max=5)
            r1[outside] = 1.0
            m.clip(min=0, max=2)
            m[outside] = 0
            return torch.stack((m * torch.cos(p), m * torch.sin(p), r1)).float()

        dict_values = {x[0]: x[1] for x in values}

        def makeDS(path, transforms):
            return CartUSDataset(
                BrainwebSlices(path, cuts=cuts, what=("mask", "classes", "rpd", "ipd", "r1",), maskval=(0, 0, 1.0), transforms=transforms, classes=dict_values),
                circular_csm,
                # partial(simple_csm, ny=(2, None), nx=(min(8,Ncoils), None), rand_pos=True),
                partial(cartesian_mask, acceleration=acceleration, center_n=center_n),
                Nd=self._q.nOut,
                Ncoils=Ncoils,
                noise_strength=noise_strength,
                augment_fun=augment_values if value_augment else None,
            )

        self._dsTrain = makeDS(path / "train", transforms)
        self._dsVal = makeDS(path / "val", transforms)
        self._dsTest = makeDS(path / "test", transformsTest)
        super()._cacheDS(("_dsVal", "_dsTest"))
        self.returnIndices = (slice(0, 2, None), slice(3, None, None))

    def getEncodingOperator(self, *args):
        return Cartesian_Operator(*args)


class T1BrainwebCartInv(T1BrainwebCartBase):
    def __init__(
        self,
        path="data/brainwebClasses/",
        batchsize=8,
        acceleration=8,
        Ncoils=8,
        ti=(0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
        noise_strength=(0, 0.05),
        cuts=(120, 120, 0, 0, 0, 0),
        center_n=8,
        Npx=192,
        values=values_t1t2pd3T,
        value_augment=True,
    ):
        self._q = T1Inversion(ti)
        super().__init__(
            path=path, batchsize=batchsize, acceleration=acceleration, Ncoils=Ncoils, noise_strength=noise_strength, cuts=cuts, center_n=center_n, Npx=Npx, values=values,
        )


class T1BrainwebCartSat(T1BrainwebCartBase):
    def __init__(
        self,
        path="data/brainwebClasses/",
        batchsize=8,
        acceleration=8,
        Ncoils=8,
        ti=(0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2, 8),
        noise_strength=(0, 0.05),
        cuts=(120, 120, 0, 0, 0, 0),
        center_n=8,
        Npx=192,
        values=values_t1t2pd3T,
        value_augment=True,
    ):

        self._q = T1Saturation(ti)
        super().__init__(
            path=path, batchsize=batchsize, acceleration=acceleration, Ncoils=Ncoils, noise_strength=noise_strength, cuts=cuts, center_n=center_n, Npx=Npx, values=values, value_augment=value_augment
        )
