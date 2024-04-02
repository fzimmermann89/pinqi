import h5py
import numpy as np
from pathlib import Path
import torch
import hashlib


def hashparams(hparams, ignore=()):
    h = 0
    for k, v in hparams.items():
        if k not in ignore:
            t = (k, v)
            h ^= int.from_bytes(hashlib.sha256(str(t).encode("utf-8")).digest(), "little")
    return hex(h)[-8:]


def to_hdf5(ds, path):
    def wf(_):
        import torch
        import numpy
        import random

        torch.manual_seed(0)
        numpy.random.seed(0)
        random.seed(0)

    dl = torch.utils.data.DataLoader(ds, worker_init_fn=wf, num_workers=1, batch_size=1, shuffle=False)
    with h5py.File(path, "x") as outfile:
        for i, data in enumerate(dl):
            for j, d in enumerate(data):
                outfile[f"/{i}/{j}"] = np.array(d[0])
            outfile[f"/{i}"].attrs["max"] = j
        outfile.attrs["max"] = i


class hdf5DS(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(self.path, "r") as f:
            self._len = int(f.attrs["max"]) + 1

    def __getitem__(self, idx):
        if idx >= self._len or idx < -self._len:
            raise IndexError("index out of range")
        i = self._len + idx if idx < 0 else idx
        ret = []
        with h5py.File(self.path, "r") as f:
            ds = f[f"/{i}"]
            for j in range(ds.attrs["max"] + 1):
                d = ds[f"{j}"]
                ret.append(np.array(d))
        return tuple(ret)

    def __len__(self):
        return self._len


def cacheDS(ds, path, settings, ignore=("path", "batchsize"), prefix=""):
    filename = Path(path) / f"{prefix}{hashparams(settings,ignore)}.cache"
    try:
        ds = hdf5DS(filename)
        print(f"Using Cached {filename}")
    except FileNotFoundError:
        print(f"Writing Cached dataset {filename}")
        to_hdf5(ds, filename)
        ds = hdf5DS(filename)
    return ds
