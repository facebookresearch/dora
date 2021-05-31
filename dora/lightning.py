"""
Support for PyTorch lightning. You should just replace the call
to `Trainer(...)` with `get_trainer(...)`.
"""
import inspect
import typing as tp

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer import Trainer
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


class DoraDDPPlugin(DDPPlugin):
    """DDP plugin for compatibility with Dora.
    """
    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        distrib.init(self.torch_distributed_backend)


class RestoreDoraHistory(Callback):
    """Make sure Dora history, and checkpoint state are in sync.
    """
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        history = checkpoint['dora_link_history']
        get_xp().link.update_history(history)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['dora_link_history'] = get_xp().link.history
        return checkpoint


class _ArmDoraLogger(Callback):
    # See DoraHistoryLogger hereafter. This should go away once
    # PL supports filtering per epoch metrics only.
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        self.logger._armed = True


class _DummySLURMConnector:
    # Deactivate SLURM connector because Submitit does it already,
    # and this can cost us an unfinished epoch, which we don't want!!
    def register_slurm_signal_handlers(self):
        pass


class DoraHistoryLogger(LightningLoggerBase):
    def __init__(self, link):
        super().__init__()
        self.link = link
        # Some metrics are per step, some per epoch, I want only the per epoch.
        # At the moment this is not supported by PL, so I'll "arm" the logger
        # on the end epoch event, so that the next call to log_metrics
        # is actually going through.
        self._armed = False

    def log_metrics(self, metrics, step):
        if self._armed:
            self._armed = False
            self.link.push_metrics(metrics)

    @property
    def save_dir(self):
        return get_xp().folder

    @property
    def name(self):
        return "DoraHistoryLogger"


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

    # Convert all to kwargs
    kwargs = inspect.getcallargs(Trainer.__init__, *args, **kwargs)

    plugins = kwargs.pop("plugins", [])
    env = DoraEnvironment()

    gpus = 1
    if env.world_size() > 1:
        # Dora always use all available GPUs, either through `-d` flag locally,
        # or through Slurm (that will mask the other ones).
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
        gpus = len(devices)
        ddp = DoraDDPPlugin(cluster_environment=env, parallel_devices=devices)
        plugins += [env, ddp]
    kwargs['plugins'] = plugins

    callbacks = kwargs.pop("callbacks", [])
    callbacks.append(RestoreDoraHistory())
    callbacks.append(_ArmDoraLogger())
    kwargs['callbacks'] = callbacks

    if 'gpus' in kwargs:
        raise RuntimeError("You cannot specify the number of GPUs, as this is provided by Dora.")
    if 'num_nodes' in kwargs:
        raise RuntimeError("You cannot specify the number of nodes, as this is provided by Dora.")

    kwargs['gpus'] = gpus
    kwargs['num_nodes'] = env.num_nodes

    if 'default_root_dir' not in kwargs:
        kwargs['default_root_dir'] = get_xp().folder

    if add_dora_logger:
        logger = kwargs.pop('logger', [])
        if not isinstance(logger, tp.Iterable):
            logger = [logger]
        dora_logger = DoraHistoryLogger()
        kwargs['callbacks'].append(_ArmDoraLogger(dora_logger))
        logger.append(dora_logger)

    trainer = Trainer(**kwargs)

    if no_unfinished_epochs:
        trainer.slurm_connector = _DummySLURMConnector()
