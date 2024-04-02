import E2E
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from typing import Set, Dict, Optional, Union, List
import warnings, logging
import os, sys
from datetime import timedelta
from pathlib import Path

os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyOTZmZWQxZi00MDg5LTQ4MWItOTBmZi1lNGE4NzQ0NTlmMTcifQ=="
os.environ["NEPTUNE_PROJECT"] = "fzimmermann/e2e"


def load_checkpoint(path):
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.LightningModule, "model", subclass_mode=True)
    parser.add_lightning_class_args(pl.LightningDataModule, "data", subclass_mode=True)
    parser.add_lightning_class_args(ProblemSolutionTrainer, "trainer")
    ProblemSolutionCLI.add_arguments_to_parser(parser)
    path = Path(f"lightning_logs/version_{version}/")
    res = parser.instantiate_classes(parser.parse_path(str(path / "config.yaml"), _skip_check=True))
    model = res["model"]
    problem = res["data"]
    filename = list((path / "checkpoints").glob("*.ckpt"))
    if filename:
        chkpt = torch.load(filename[-1], map_location=torch.device("cpu"))
        model.load_state_dict(chkpt["state_dict"])
    else:
        raise FileNotFoundError("no checkpoint")
    return dict(model=model, problem=problem, hyper_parameters={**chkpt["hyper_parameters"], "model": type(model).__name__})


class ProblemSolutionCLI(LightningCLI):
    def parse_arguments(self, parser: LightningArgumentParser, args) -> None:
        commands = list(self.subcommands().keys())
        if len(sys.argv) < 2 or sys.argv[1] not in commands:
            sys.argv.insert(1, commands[0])
        self.config = parser.parse_args(args)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        # The other commands are currently not implemented
        return {"fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"}}

    @classmethod
    def add_arguments_to_parser(cls, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.link_arguments("data", "model.init_args.Problem", apply_on="instantiate", compute_fn=cls.Identity)
        parser.set_defaults({"trainer.max_steps": 100000, "trainer.max_epochs": 500, "trainer.accelerator": "gpu", "trainer.devices": "0,", "checkpoint.every_n_epochs": 5})
        parser.add_optimizer_args = lambda *x: None
        parser.add_lr_scheduler_args = lambda *x: None

    @staticmethod
    def Identity(x):
        return x


class ProblemSolutionTrainer(LightningTrainer):
    def __init__(
        self,
        fast_dev_run=False,
        # logger: Union[Logger, Iterable[Logger], bool] = True,
        enable_checkpointing: bool = True,
        default_root_dir: Optional[str] = None,
        # gradient_clip_val: Optional[Union[int, float]] = None,
        # gradient_clip_algorithm: Optional[str] = None,
        num_nodes: int = 1,
        num_processes: Optional[int] = None,  # TODO: Remove in 2.0
        devices: Optional[Union[List[int], str, int]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        enable_progress_bar: bool = True,
        # overfit_batches: Union[int, float] = 0.0,
        # track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: Optional[int] = 1,
        # fast_dev_run: Union[int, bool] = False,
        # accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
        max_epochs: Optional[int] = None,
        max_steps: int = -1,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        val_check_interval: Optional[Union[int, float]] = None,
        accelerator: Optional[str] = None,
        # strategy: Optional[str] = None,
        # sync_batchnorm: bool = False,
        enable_model_summary: bool = False,
        # weights_save_path: Optional[str] = None,  # TODO: Remove in 1.8
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        # profiler: Optional[str] = None,
        # benchmark: Optional[bool] = None,
        deterministic: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        # auto_lr_find: Union[bool, str] = False,
        # replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,
        # auto_scale_batch_size: Union[str, bool] = False,
        # move_metrics_to_cpu: bool = False,
        callbacks=None,
    ):
        args = {k: v for (k, v) in locals().items() if k[0] != "_" and k != "self"}
        super().__init__(**args)


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    cli = ProblemSolutionCLI(E2E.solutions.base.Solution, E2E.problems.base.Problem, subclass_mode_model=True, subclass_mode_data=True, trainer_class=ProblemSolutionTrainer)
    # import pdb;pdb.set_trace()
