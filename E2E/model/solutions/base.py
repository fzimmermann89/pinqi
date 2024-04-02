import torch
import pytorch_lightning as pl
import neptune.new as neptune
import abc

from .validation import validationE2E as validator


class Solution(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, Problem):
        super().__init__()
        self.save_hyperparameters(ignore=["Problem"])
        self.q = Problem.q
        self.Problem = Problem
        self.getEncodingOperator = Problem.getEncodingOperator
        self.automatic_optimization = False
        self.run = None
        self.validator = None
        self.additional_info = {}
        self.losses = []

    def on_train_start(self):
        opt = self.optimizers()
        print(18 * "\b", end="")
        logdir = self.trainer.log_dir
        self.run = neptune.init(source_files=["train.py", "test.ipynb", "E2E/**/*.py", f"{logdir}/*.yaml"])
        self.run["parameters"] = dict(logdir=logdir, **self.additional_info, **self.hparams, **self.Problem.hparams, Problem=self.Problem.__class__.__name__, Solution=self.__class__.__name__)
        self.validator = validator(self.trainer.val_dataloaders[0], self.run, self, optimizer=opt)
        print()
        self.losses = []

    def on_train_end(self):
        if self.run is not None:
            self.run.stop()

    def training_step_end(self, loss):
        self.losses.append(loss.item())
        self.log("loss", loss, prog_bar=True)

    def on_validation_model_eval(self):
        if self.validator is not None:
            l = self.validator((self.losses,))
        self.losses = []

    def validation_step(self, *args):
        pass

    def training_step(self, batch, batch_idx):
        try:
            opt, optPre = self.optimizers()
        except TypeError:
            opt, optPre = self.optimizers(), None
        try:
            sched, schedPre = self.lr_schedulers()
        except TypeError:
            sched, schedPre = self.lr_schedulers(), None
        if "pretrain_length" in self.hparams and self.current_epoch < self.hparams.pretrain_length and hasattr(self, "pretrain_step"):
            return self.pretrain_step(batch, optPre, schedPre, batch_idx)
        else:
            return self.train_step(batch, opt, sched, batch_idx)

    def __del__(self):
        if self.run is not None:
            self.run.stop()

    @abc.abstractmethod
    def forward(self, *args):
        pass

    def train_step(self, batch, opt, sched, batch_idx):
        pass
