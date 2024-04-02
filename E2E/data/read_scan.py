#%%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import PTBRecon
import argparse
import h5py
import re
import matplotlib


# %matplotlib widget
# import sys
# sys.argv=['','data/2022-07-14_debug/meas_MID141*']
# %%

parser = argparse.ArgumentParser()
parser.add_argument("inputpattern", type=str, help="Example: data/2022-07-14_debug/meas_MID144*")
parser.add_argument("--doradial", default=False, action="store_true")
args = parser.parse_args()
folder = next(Path(".").glob(args.inputpattern))
ndyns = int(re.search(r"([0-9]+)vals", str(folder)).groups()[0])  # this should be read from the sequence...

# %% load data
dat = next(folder.glob("*.dat"))
seq = next(folder.glob("*.seq"))

mr = PTBRecon.MRScan(str(dat))
mr.Pars.Flags.Verbose = 0
mr.ReadHdr(mr)
mr.ReadData(mr)
mr.Pars.Recon.PulseqPathname = str(seq.parent)
mr.Pars.Recon.PulseqFilename = str(seq.name)

if args.doradial:
    mr.ReadPulseq(mr, overwrite={"repetition": 1, "sampling_scheme": "radial"})
else:
    mr.ReadPulseq(mr, overwrite={"repetition": 1})

mr.SortData(mr)
mr.CalcTraj(mr)
mr.Pars.Recon.SplitDynSlWnd = 0
mr.Pars.Recon.SplitDynNum = ndyns
mr.SplitDyn(mr)

# %% csm kmask/kdata
mr.CalcDcf(mr)
mr.Pars.Recon.CsmMode = "inati"
mr.CalcCsm(mr)
csm = np.copy(np.squeeze(mr.Pars.Recon.Csm))

# %% get kmask/kdata
fftshift = False
acceleration = mr.Pars.Recon.PulseqHead["oversampling"]
ktraj = mr.Pars.Recon.KTraj
kftdim = mr.Pars.Encoding.KFTDims
rows = np.squeeze(np.round(ktraj[0, :, 0, 1] * kftdim[1] + mr.Pars.Encoding.Idx.KCtr[1, 0]).astype(np.int64))
dyn = np.broadcast_to(np.arange(ndyns)[None, :], (len(rows), rows.shape[1]))
kdata = np.zeros((*kftdim[:2], mr.Data.K.shape[6], mr.Data.K.shape[3]), dtype=np.complex64)
kdata[:, rows, dyn] = np.squeeze(mr.Data.K).swapaxes(-1, -2)
kdata = np.moveaxis(kdata, (-1, -2), (1, 0))
kmask = np.zeros((rows.shape[1], kftdim[1]))
kmask[dyn, rows] = 1
if fftshift:
    kmask = np.fft.fftshift(kmask, -1)
    kdata = np.fft.fftshift(kdata, -1)
kmask = np.broadcast_to(kmask[:, None, :], (ndyns, *kftdim[:-1]))

# %% reconstruct data
# just use one csm
# mr.Pars.Recon.Csm[:,:,:,:,:,:,:-1] = mr.Pars.Recon.Csm[:,:,:,:,:,:,-1,None,...] #last
# mr.Pars.Recon.Csm[:,:,:,:,:,:,1:] = mr.Pars.Recon.Csm[:,:,:,:,:,:,0,None,...] #first
# use mean of csms
mr.Pars.Recon.Csm[:] = np.mean(mr.Pars.Recon.Csm, axis=6, keepdims=True)
mr.Pars.Recon.ReconType = "fft"
mr.ReconData(mr)
mr.Pars.Recon.CoilCombineType = "sw"
mr.CombineCoils(mr)

# %%
img = np.squeeze(mr.Data.I)
# img = np.fliplr(np.rot90(img, -1, axes=(0, 1)))

# %%
for i, f in enumerate(plt.subplots(3, 3, figsize=(16, 12))[1].ravel()):
    if i >= ndyns:
        break
    p = f.matshow(np.abs(img[..., i]), vmax=0.04, vmin=0)
    f.set_title(i)
    plt.colorbar(p, ax=f)
plt.savefig(folder / f'img{"rad" if args.doradial else ""}.png', dpi=144)
plt.close()

# %%
for i, f in enumerate(plt.subplots(3, 3, figsize=(16, 12))[1].ravel()):
    if i >= ndyns:
        break
    p = f.matshow(np.log(np.abs(kdata[i, 0])))
    f.set_title(i)
    plt.colorbar(p, ax=f)
plt.savefig(folder / "k.png", dpi=144)
plt.close()

# %%
with h5py.File(folder / "data.h5", "w") as f:
    f["csm"] = csm
    f["kdata"] = kdata
    f["kmask"] = kmask
    f.attrs["seq_name"] = str(seq)
    f.attrs["dat_name"] = str(dat)
print(f"saved to {folder}")

# %%
