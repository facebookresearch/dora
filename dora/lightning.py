# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Support for PyTorch lightning. You should just replace the call
to `Trainer(...)` with `get_trainer(...)`.
"""
import argparse
import functools
import inspect
import os
import typing as tp

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.plugins import TrainingTypePluginsRegistry
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.argparse import from_argparse_args
import torch

from . import distrib
from .xp import get_xp, is_xp


class DoraEnvironment(ClusterEnvironment):
    def __init__(self):
        super().__init__()
        self.spec = distrib.get_distrib_spec()

    def creates_children(self) -> bool:
        return True

    def master_address(self) -> str:
        return ""

    def master_port(self) -> int:
        assert False

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


@TrainingTypePluginsRegistry.register("dora_ddp")
class DoraDDPPlugin(DDPPlugin):
    """DDP plugin for compatibility with Dora.
    """
    def init_ddp_connection(self, global_rank: tp.Optional[int] = None,
                            world_size: tp.Optional[int] = None) -> None:
        distrib.init(self.torch_distributed_backend)


class RestoreDoraHistory(Callback):
    """Make sure Dora history, and checkpoint state are in sync.
    """
    def __init__(self):
        self.link = get_xp().link

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        history = checkpoint['dora_link_history']
        self.link.update_history(history)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['dora_link_history'] = self.link.history
        return checkpoint


class _ArmDoraLogger(Callback):
    # Some metrics are per step, some per epoch, I want only the per epoch.
    # At the moment this is not supported by PL, so I manually trigger the logger
    # save on some events that are registered with this callback.
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self._first = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self._first:
            self._first = False
            return
        self.logger._push()

    def on_train_end(self, trainer, pl_module):
        self.logger._push()

    def on_test_end(self, trainer, pl_module):
        self.logger._repush()


class _DummySLURMConnector:
    # Deactivate SLURM connector because Submitit does it already,
    # and this can cost us an unfinished epoch, which we don't want!!
    def register_slurm_signal_handlers(self):
        pass


class DoraHistoryLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.link = get_xp().link
        self.folder = get_xp().folder
        self._metrics = {}

    def log_metrics(self, metrics, step):
        self._metrics.update(metrics)

    def _push(self):
        self.link.push_metrics(self._metrics)
        self._metrics = {}

    def _repush(self):
        history = self.link.history
        history[-1].update(self._metrics)
        self.link.update_history(history)
        self._metrics = {}

    @property
    def save_dir(self):
        return self.folder

    @property
    def name(self):
        return "DoraHistoryLogger"

    def experiment(self) -> tp.Any:
        """Return the experiment object associated with this logger."""
        pass

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        pass

    @property
    def version(self) -> int:
        """Return the experiment version."""
        return 0


def get_trainer(*args, add_dora_logger=True, no_unfinished_epochs=True, **kwargs):
    """Return a PL trainer, adding the necessary glue code to make everything works.
    The arguments are exactly the same as for `pytorch_lightning.trainer.Trainer`,
    with a few extras documented after.

    ..note:: You should not pass `gpus=` or `num_nodes=` arguments as those will be filled by Dora.

    Args:
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
        plugins += [env, 'dora_ddp']
    kwargs['plugins'] = plugins

    callbacks = kwargs.pop("callbacks", [])
    callbacks.append(RestoreDoraHistory())
    kwargs['callbacks'] = callbacks

    if kwargs['gpus'] is not None:
        raise RuntimeError("You cannot specify the number of GPUs, as this is provided by Dora.")
    if kwargs['num_nodes'] != 1:
        raise RuntimeError("You cannot specify the number of nodes, as this is provided by Dora.")

    kwargs['gpus'] = gpus
    kwargs['num_nodes'] = env.spec.num_nodes
    kwargs['default_root_dir'] = get_xp().folder

    if add_dora_logger:
        logger = kwargs['logger']
        if logger is True:
            version = os.environ.get('PL_EXP_VERSION')
            if version is None:
                version = os.environ.get('SLURM_JOB_ID')
            # Create default logger as in PL logger_connector.py
            logger = TensorBoardLogger(
                save_dir=get_xp().folder, version=version, name='lightning_logs')
        if not isinstance(logger, tp.Iterable):
            logger = [logger]
        dora_logger = DoraHistoryLogger()
        kwargs['callbacks'].append(_ArmDoraLogger(dora_logger))
        logger.append(dora_logger)
        kwargs['logger'] = logger

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
