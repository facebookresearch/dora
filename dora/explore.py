from collections import OrderedDict
from copy import deepcopy
import typing as tp

import treetable as tt

from .conf import SlurmConfig
from .shep import Shepherd


class Launcher:
    def __init__(self, shepherd: Shepherd, herd: OrderedDict,
                 slurm: SlurmConfig, args: tp.Optional[tp.List[tp.Any]] = None):
        self.shepherd = shepherd
        self.herd = herd
        self.slurm = slurm
        self.args = self.args or []

    def bind(self, *args, **kwargs):
        new = deepcopy(self)
        return new.bind(*args, **kwargs)

    def bind_(self, *args, **kwargs):
        self.args += args
        for key, value in kwargs.items():
            assert hasattr(self.slurm, key)
            setattr(self.slurm, key, value)
        return self

    def __call__(self, *args, **kwargs):
        launcher = self.bind(*args, **kwargs)
        argv = self.shepherd.main.merge_args(launcher.args)
        sheep = self.shepherd.get_sheep(argv)
        self.herd[sheep.run.sig] = (sheep, launcher.slurm)


class Explorer:
    def __call__(self, launcher: Launcher):
        raise NotImplementedError()

    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table.
        """
        return tt.group("Metrics", [])

    def get_colors(self):
        return ["0", "38;5;245"]
