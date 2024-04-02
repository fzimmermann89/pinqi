import abc
import pytorch_lightning as pl
import torch
from ...util import cacheDS, filterDS


class Problem(pl.LightningDataModule, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.returnIndices = None

    @property
    def q(self):
        return self._q

    def val_dataloader(self):
        return torch.utils.data.DataLoader(filterDS(self._dsVal, self.returnIndices), batch_size=8, shuffle=False, num_workers=4, drop_last=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(filterDS(self._dsTrain, self.returnIndices), batch_size=self.hparams.batchsize, shuffle=True, num_workers=4, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(filterDS(self._dsTest,self. returnIndices), batch_size=8, shuffle=False, num_workers=4, drop_last=True)

    def _cacheDS(self, which, ignorekeys=("path", "batchsize")):
        for name in which:
            ds = getattr(self, name)
            setattr(self, name, cacheDS(ds, path=self.hparams.path, settings=self.hparams, ignore=ignorekeys, prefix=type(self).__name__ + name))
    
    @abc.abstractmethod
    def getEncodingOperator(self, *args):
        pass
