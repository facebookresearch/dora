from collections import OrderedDict
import typing as tp

from treetable.table import _Node
import treetable as tt

from .conf import SlurmConfig
from .shep import Shepherd


class Launcher:
    def __init__(self, shepherd: Shepherd, slurm: SlurmConfig, herd: OrderedDict,
                 argv: tp.List[str] = []):
        self._shepherd = shepherd
        self._main = self._shepherd.main
        self._herd = herd
        self._slurm = slurm
        self._argv = list(argv)

    def copy(self):
        return Launcher(self._shepherd, self._slurm, self._herd, self._argv)

    def bind(self, *args, **kwargs):
        new = self.copy()
        return new.bind_(*args, **kwargs)

    def bind_(self, *args, **kwargs):
        for arg in args:
            self._argv += self._main.value_to_argv(arg)
        self._argv += self._main.value_to_argv(kwargs)
        return self

    def slurm(self, **kwargs):
        new = self.copy()
        return new.slurm_(**kwargs)

    def slurm_(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self._slurm, key):
                raise AttributeError(f"Invalid Slurm config {key}")
            setattr(self._slurm, key, value)
        return self

    def __call__(self, *args, **kwargs):
        launcher = self.bind(*args, **kwargs)
        sheep = self._shepherd.get_sheep_from_argv(launcher._argv)
        self._herd[sheep.xp.sig] = (sheep, launcher._slurm)
        return sheep


Explore = tp.Callable[[Launcher], None]


class Explorer:
    def __init__(self, explore: Explore):
        self.explore = explore

    def __call__(self, launcher: Launcher):
        self.explore(launcher)

    def get_grid_metrics(self) -> tp.List[_Node]:
        """Return the metrics that should be displayed in the tracking table.
        """
        return []

    def get_grid_meta(self) -> tp.List[_Node]:
        """Returns the list of Meta information to display for each XP/job.
        """
        return [
            tt.leaf("index", align=">"),
            tt.leaf("name"),
            tt.leaf("state"),
            tt.leaf("sig"),
            tt.leaf("sid", align=">"),
            tt.leaf("epoch", align=">"),
        ]

    def get_colors(self):
        return ["0", "38;5;245"]
