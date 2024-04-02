import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as snd
import torch
from torch import tensor, Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from ..util import cutnan, trunc_norm


@dataclass
class t1t2pd:
    t1min: float
    t1max: float
    t2min: float
    t2max: float
    pdmin: float
    pdmax: float

    @property
    def r1(self):
        return 1000 / random.uniform(self.t1min, self.t1max)

    @property
    def r2(self):
        return 1000 / random.uniform(self.t2min, self.t2max)

    @property
    def rpd(self):
        return random.uniform(self.pdmin, self.pdmax)
    
    @property
    def ipd(self):
        return random.uniform(-self.pdmax/10, self.pdmax/10)
    
    @classmethod
    def dictfromdict(cls,dict):
        return {key:cls(*vals) for key, vals in dict.items()}


values_t1t2pd3T = [
               # t1min t1max t2min t2max pdmin pdmax
        ("gry", (1200, 2000, 80, 120, 0.7, 1.0)),
        ("wht", (800, 1500, 60, 100, 0.50, 0.9)),
        ("csf", (2000, 4000, 1300, 2000, 0.9, 1.0)),
        ("mrw", (400, 600, 60, 100, 0.7, 1.0)),
        ("dura", (2000, 2800, 200, 500, 0.9, 1.0)),
        ("fat", (300, 500, 60, 100, 0.9, 1.0)),
        ("fat2", (400, 600, 60, 100, 0.6, 0.9)),
        ("mus", (1200, 1500, 40, 60, 0.9, 1.0)),
        ("m-s", (500, 900, 300, 500, 0.9, 1)),
        ("ves", (1700, 2100, 200, 400, 0.8, 1)),
]


class BrainwebSlices(Dataset):
    def __init__(
        self,
        folder,
        cuts=(0, 0, 0, 0, 0, 0),
        axis=0,
        step=1,
        transforms=(
            (
                T.RandomAffine(5, translate=(0.1, 0.1), scale=(0.65, 0.85), fill=0.0, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((256, 256)),
                T.RandomHorizontalFlip(),
                T.RandomHorizontalFlip(),
            ),
            (None,),
         ),
        classes=None,
        what=("pd", "r1", "r2"),
        maskval=None,
    ):
        if classes is None:
            self.classes = t1t2pd.dictfromdict(values_t1t2pd3T)
        else:
            self.classes = t1t2pd.dictfromdict(classes)
        self._cuts = cuts
        self._axis = axis
        self._step = step
        files = []
        ns = [0]
        for fn in Path(folder).glob("*.h5"):
            try:
                with h5py.File(fn) as f:
                    ns.append(
                        (f["classes"].shape[self._axis]) - (self._cuts[self._axis * 2] + self._cuts[self._axis * 2 + 1])
                    )
                    files.append(fn)
            except:
                pass
        self._files = tuple(files)
        self._ns = np.cumsum(ns)
        self._transforms = (T.Compose(transforms[0]), T.Compose(transforms[1]))
        self.what = what
        self._maskval = maskval

    def __len__(self):
        return self._ns[-1] // self._step

    def __getitem__(self, index):
        if index * self._step >= self._ns[-1]:
            raise IndexError
        elif index < 0:
            index = self._ns[-1] + index * self._step
        else:
            index = index * self._step
        fileid = np.searchsorted(self._ns, index, "right") - 1
        sliceid = index - self._ns[fileid] + self._cuts[self._axis * 2]
        with h5py.File(
            self._files[fileid],
        ) as f:
            where = [slice(self._cuts[2 * i], f["classes"].shape[i] - self._cuts[2 * i + 1]) for i in range(3)] + [
                slice(None)
            ]
            where[self._axis] = sliceid
            data = np.array(f["classes"][tuple(where)], dtype=np.uint8)
            classnames = tuple(f.attrs["classnames"])
            mask = data.sum(-1) > 150  # .astype(bool)
            ret = np.zeros((len(self.what),) + data.shape[:-1], dtype=np.float32)
            maskval = [*self._maskval] if self._maskval is not None else None
            maskindex = None
            for i, el in enumerate(self.what):
                if el == "r1":
                    ret[i] = np.dot(data, (np.array([self.classes[k].r1 for k in classnames])) / 255)
                    ret[i, ~mask] = np.nan
                elif el == "r2":
                    ret[i] = np.dot(data, (np.array([self.classes[k].r2 for k in classnames])) / 255)
                    ret[i, ~mask] = np.nan
                elif el == "rpd" or el == "pd":
                    ret[i] = np.dot(data, (np.array([self.classes[k].rpd for k in classnames])) / 255)
                    ret[i, ~mask] = np.nan
                elif el == "ipd":
                    ret[i] = np.dot(data, (np.array([self.classes[k].ipd for k in classnames])) / 255)
                    ret[i, ~mask] = np.nan
                elif el == "t1":
                    r = np.dot(data, (np.array([self.classes[k].r1 for k in classnames])) / 255)
                    ret[i] = np.reciprocal(r, out=np.zeros_like(r), where=mask)
                    ret[i, ~mask] = np.nan
                elif el == "t2":
                    r = np.dot(data, (np.array([self.classes[k].r2 for k in classnames])) / 255)
                    ret[i] = np.reciprocal(r, out=np.zeros_like(r), where=mask)
                    ret[i, ~mask] = np.nan
                elif el == "mask":
                    maskindex = i
                    if maskval is not None:
                        maskval.insert(i, 0)
                elif el == "classes":
                    ret[i] = np.argmax(data, -1)
                    ret[i, ~mask] = np.nan
                    if maskval is not None:
                        maskval.insert(i, -1)
                else:
                    raise NotImplementedError(
                        f"what=({el},) is not implemented. Each element of what shall be one of r1, r2, (i/r)pd, t1, t2, classes or mask"
                    )
        ret = torch.from_numpy(cutnan(ret))
        ret = self._transforms[0](ret)
        mask = torch.any(torch.isnan(ret), 0) | torch.all(ret == 0.0, 0)
        if maskval is not None:
            for i, val in enumerate(maskval):
                ret[i, mask] = val
        for i, el in enumerate(self.what):
            if el == "classes":
                rounded = np.round(ret[i])
                ret[i, np.abs(ret[i] - rounded) > 0.1] = -1
                ret[i, mask] = -1
            elif el == "mask":
                ret[i, 1:-1, 1:-1] = ~(torch.nn.functional.conv2d(mask[None, None, ...].float(), torch.ones(1, 1, 3, 3))[0, 0] > 0)
            else:
                ret[i] = self._transforms[1](ret[i].unsqueeze(0)).squeeze(0)

        return ret
