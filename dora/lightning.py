# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Support for PyTorch lightning. You should just replace the call
to `Trainer(...)` with `get_trainer(...)`.
For using `dora.log.LogProgress` as a progress bar with PL, see `PLLogProgress`.
"""
import functools
import inspect
import os
import typing as tp

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.argparse import from_argparse_args
import torch

from . import distrib
from .xp import get_xp, is_xp
from .log import bold, LogProgress


def _filter_metrics(metrics: tp.Dict[str, tp.Any], epoch: bool = True):
    """Filters metrics before formatting, in particular to remove the `_step` or `_epoch`
    suffix. This will also convert torch tensors to float.
    Args:
        metrics: dict given by PL.
        epoch: if True, keep only epoch level metrics, otherwise, keep only step level metrics.
    """
    out = {}
    for key, value in metrics.items():
        if epoch and key.endswith('_step'):
            continue
        if not epoch and key.endswith('_epoch'):
            continue
        if key.endswith('_step') or key.endswith('_epoch'):
            key = key.rsplit('_', 1)[0]
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        out[key] = value
    return out


class DoraEnvironment(ClusterEnvironment):
    def __init__(self):
        super().__init__()
        self.spec = distrib.get_distrib_spec()
        distrib.set_distrib_env()

    def creates_children(self) -> bool:
        return True

    @property
    def creates_processes_externally(self) -> bool:
        return True

    def world_size(self) -> int:
        return self.spec.world_size

    def set_world_size(self, size: int) -> None:
        pass

    def global_rank(self) -> int:
        return self.spec.rank

    def set_global_rank(self, rank: int) -> None:
        pass

    def local_rank(self) -> int:
        return self.spec.local_rank

    def node_rank(self) -> int:
        return self.spec.node_rank

    @staticmethod
    def detect() -> bool:
        return False

    @property
    def main_address(self) -> str:
        return os.environ["MAIN_ADDR"]

    @property
    def main_port(self) -> int:
        return int(os.environ["MAIN_PORT"])


class DoraCheckpointSync(Callback):
    """Make sure Dora history, and checkpoint state are in sync.
    """
    def __init__(self):
        self.xp = get_xp()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        history = checkpoint['dora_link_history']
        self.xp.link.update_history(history)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['dora_link_history'] = self.xp.link.history
        checkpoint['dora_sig'] = self.xp.sig
        checkpoint['dora_cfg'] = self.xp.cfg
        return checkpoint


class DoraHistoryLogger(Callback):
    """Save metrics to Dora using the XP link.
    """
    def __init__(self):
        super().__init__()
        self.link = get_xp().link

    def on_fit_start(self, trainer, pl_module):
        self._first_valid = True

    def on_train_epoch_start(self, trainer, pl_module):
        self._first_valid = False

    def on_epoch_end(self, trainer, pl_module):
        if self._first_valid:
            # We ignore the first fake epoch of PL that only does a few valid batches.
            return
        metrics = trainer.logged_metrics
        metrics = _filter_metrics(metrics, epoch=True)
        self.link.push_metrics(metrics)


class _DummySLURMConnector:
    # Deactivate SLURM connector because Submitit does it already,
    # and this can cost us an unfinished epoch, which we don't want!!
    def register_slurm_signal_handlers(self):
        pass


def get_trainer(*args, auto_resume=True, add_dora_logger=True, no_unfinished_epochs=True,
                **kwargs):
    """Return a PL trainer, adding the necessary glue code to make everything works.
    The arguments are exactly the same as for `pytorch_lightning.trainer.Trainer`,
    with a few extras documented after.

    ..note:: You should not pass `gpus=` or `num_nodes=` arguments as those will be filled by Dora.

    Args:
        auto_resume (bool): if True, automatically resume previous checkpoints.
            You are still responsible for creating the `ModelCheckpoint` callback,
            this only handles the `resume_from_checkpoint` part.
        add_dora_logger (bool): if True, adds a Dora Logger to automatically
            forward the metrics (those logged with per_epoch=True), otherwise
            pushing metrics will be up to you.
        no_unfinished_epochs (bool): if True, deactivates SLURM signal handling
            by PL, which can result in half finished epoch with each interruption.
            It is recommended to instead dump a checkpoint every epoch and resume
            from that one so that training is reliable.

    """
    if not is_xp():
        raise RuntimeError("This can only be called from inside a Dora XP.")

    # Convert all to kwargs, add [None] dummy for self which is missing.
    init = Trainer.__init__
    while hasattr(init, '__wrapped__'):
        init = init.__wrapped__
    kwargs = inspect.getcallargs(init, [None] + list(args), **kwargs)
    del kwargs['self']

    plugins = kwargs.pop("plugins") or []
    env = DoraEnvironment()
    gpus = min(torch.cuda.device_count(), env.world_size())
    if env.world_size() > 1:
        plugins += [env, 'ddp']
    kwargs['plugins'] = plugins

    callbacks = kwargs.pop("callbacks", None) or []
    callbacks.append(DoraCheckpointSync())
    kwargs['callbacks'] = callbacks

    if kwargs['gpus'] is not None:
        raise RuntimeError("You cannot specify the number of GPUs, as this is provided by Dora.")
    if kwargs['num_nodes'] != 1:
        raise RuntimeError("You cannot specify the number of nodes, as this is provided by Dora.")

    kwargs['gpus'] = gpus
    kwargs['num_nodes'] = env.spec.num_nodes
    kwargs['default_root_dir'] = get_xp().folder

    if add_dora_logger:
        kwargs['callbacks'].append(DoraHistoryLogger())

    resume_from_checkpoint = kwargs.get('resume_from_checkpoint')
    if auto_resume and resume_from_checkpoint is None:
        last = get_xp().folder / 'last.ckpt'
        if last.is_file():
            resume = str(last)
        else:
            resume = None
        kwargs['resume_from_checkpoint'] = resume
    trainer = Trainer(**kwargs)

    if no_unfinished_epochs:
        trainer.slurm_connector = _DummySLURMConnector()

    return trainer


class _Intercept:
    @functools.wraps(Trainer.__init__)
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def trainer_from_argparse_args(args, **kwargs):
    intercept = from_argparse_args(_Intercept, args, **kwargs)
    return get_trainer(*intercept.args, **intercept.kwargs)


class PLLogProgress(ProgressBarBase):
    """`dora.log.LogProgress` support for Pytorch-Lightning.


    """

    def __init__(self, logger, **kwargs) -> None:
        super().__init__()  # don't forget this :)
        self.logger = logger
        self.kwargs = kwargs
        self._pl_module: tp.Optional[LightningModule] = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        self._pl_module = pl_module
        self._replay_history: tp.List[tp.Any] = []

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self._in_train = False
        self._first_valid = True

    @property
    def pl_module(self) -> LightningModule:
        assert self._pl_module is not None
        return self._pl_module

    def format_metrics(self, metrics: tp.Dict[str, tp.Any],
                       stage: str, epoch: bool = False):
        """Default method to format metrics for displaying in the progress bar.
        To customize, you can define a `format_metrics()` method on your
        Lightning module.

        Args:
            metrics: dict of metrics given by PL.
            stage: "train" or "valid".
            epoch: if True, provided metrics are for the end of epoch summary.
        """
        out = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                out[key] = format(value, '.5f')
        return out

    @property
    def _format_metrics(self):
        return getattr(self.pl_module, 'format_metrics', self.format_metrics)

    def _on_epoch_start(self, stage):
        self.logger.info("-" * 70)
        self.logger.info("Training..." if stage == "train" else "Validating...")
        name = stage.capitalize() + f" | Epoch {self.trainer.current_epoch + 1}"
        if stage == "train":
            total = int(self.total_train_batches)
        elif stage == "valid":
            total = int(self.total_val_batches)
        else:
            raise RuntimeError(f"Invalid stage {stage}")

        loader = range(total)
        self.logprog = LogProgress(self.logger, loader, total=total, name=name, **self.kwargs)
        iter(self.logprog)

    def on_train_epoch_start(self, trainer, pl_module):
        self._on_epoch_start("train")
        self._in_train = True
        self._first_valid = False
        return super().on_train_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._on_epoch_start("valid")
        return super().on_validation_epoch_start(trainer, pl_module)

    def _on_batch_end(self, stage):
        metrics = self.get_metrics(self.trainer, self.pl_module)
        metrics = _filter_metrics(metrics, epoch=False)
        formatted = self._format_metrics(metrics, stage, epoch=False)
        self.logprog.update(**formatted)
        next(self.logprog)

    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        self._on_batch_end("train")

    def on_validation_batch_end(self, *args, **kwargs):
        super().on_validation_batch_end(*args, **kwargs)
        self._on_batch_end("valid")

    def _on_stage_end(self, stage):
        if stage == "train":
            # dirty hack as we might not yet be at the end of the epoch.
            metrics = self.trainer.fit_loop.epoch_loop._results.metrics(False)["log"]
        else:
            metrics = self.trainer.fit_loop.epoch_loop.val_loop._results.metrics(False)["log"]
        metrics = _filter_metrics(metrics, epoch=False)
        self._show_epoch_summary(stage, self.trainer.current_epoch, metrics)

    def _show_epoch_summary(self, stage, epoch, metrics):
        self._replay_history.append((stage, epoch, metrics))
        formatted = self._format_metrics(metrics, stage, epoch=True)
        name = stage.capitalize()
        summary = " | ".join(
            f"{key.capitalize()}={val}" for key, val in formatted.items()
        )
        self.logger.info(bold(f"{name} Summary | End of Epoch {epoch + 1} | {summary}"))

    def on_validation_start(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        assert self._in_train or self._first_valid
        if not self._first_valid:
            self._on_stage_end("train")
            self._in_train = False

    def on_epoch_end(self, trainer, pl_module):
        super().on_epoch_end(trainer, pl_module)
        if self._in_train:
            self._on_stage_end("train")
        self._in_train = False

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        self._on_stage_end("valid")

    def disable(self):
        # we do nothing here for now. This is called by PL when using DDP,
        # but Dora already separates the stdout and stderr from the different workers.
        pass

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        replay_history = checkpoint.get('dora_replay_history', [])
        if replay_history:
            self.logger.info("Replaying past metrics...")
        for step in replay_history:
            self._show_epoch_summary(*step)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['dora_replay_history'] = self._replay_history
        return checkpoint
