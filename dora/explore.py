from collections import OrderedDict
from copy import deepcopy
import typing as tp

from .conf import SlurmConfig
from .customize import Customizations, custom
from .shep import Shepherd


class Launcher:
    def __init__(self, shepherd: Shepherd, herd: OrderedDict,
                 slurm: SlurmConfig, args=None):
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


Explore = tp.Callable[[Launcher], None]


class Explorer:
    def __init__(self, explore: Explore, custom: Customizations = custom):
        self.explore = explore
        self.custom = custom


def explorer(explore: Explore = None, *, custom: Customizations = custom):
    if explore is None:
        def _decorator(explore: Explore):
            return Explorer(explore, custom)
        return _decorator
    return Explorer(explore, custom)
