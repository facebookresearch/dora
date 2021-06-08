"""
Classes used to define a grid search.

`Launcher`: a launcher is passed to each grid search explore function,
and can be called repeatidly to schedule XPs.

`Explorer`: defines some metadata, in particular the metrics to display
with the `dora grid` command.
"""
from copy import deepcopy
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import typing as tp

from treetable.table import _Node
import treetable as tt

from .conf import SlurmConfig
from .shep import Shepherd


class ProcessException(RuntimeError):
    pass


def _process(shepherd: Shepherd, argv: tp.List[str], slurm: SlurmConfig):
    try:
        return (shepherd.get_sheep_from_argv(argv), slurm)
    except Exception as exc:
        raise ProcessException(repr(exc))


class Launcher:
    """
    A launcher is passed to the explore function and can be called repeatidly
    to schedule experiments.

    For instance:

        launcher(epochs=40)
        launcher(bs=64)

    A call to `launcher()` will schedule a new experiments, and all arguments
    have the same effect as in `Launcher.bind()`.
    """

    def __init__(self, shepherd: Shepherd, slurm: SlurmConfig, herd: OrderedDict,
                 argv: tp.List[str] = [], pool: tp.Optional[ProcessPoolExecutor] = None):
        self._shepherd = shepherd
        self._main = self._shepherd.main
        self._herd = herd
        self._slurm = deepcopy(slurm)
        self._argv = list(argv)
        self._pool = pool

    def _copy(self):
        return Launcher(self._shepherd, self._slurm, self._herd, self._argv, self._pool)

    def bind(self, *args, **kwargs):
        """
        Returns a new `Launcher` with different default XP parameters when scheduling experiments.

        Each entry in `*args` can be itself a list of dict or strings,
        or a string or a dict.

        Any string arg is considered directly as something to append to the list
        of *argv*, i.e. the command line arguments passed to the training scripts.

        A dictionary will be converted to a list of `argv`, with the specific syntax
        defined by the `main` function. For an argparse based script, a key
        value pair will be converted to `--key=value`, with some special rules
        (if the value is True, then it is converted to just `--key`).

        A list containing strings or dicts will be the concatenation
        of the argv obtained from each of its entries.

        For instance

            sub_launcher = launcher.bind(["--some_flag=5"], other_flag="test")
        """
        new = self._copy()
        return new.bind_(*args, **kwargs)

    def bind_(self, *args, **kwargs):
        """
        In-place version of `Launcher.bind()`.
        """
        for arg in args:
            self._argv += self._main.value_to_argv(arg)
        self._argv += self._main.value_to_argv(kwargs)
        return self

    def slurm(self, **kwargs):
        """
        Return a new `Launcher` with different default Slurm parameters.

        For instance

            sub_launcher = launcher.slurm(cpus_per_task=20)

        """

        new = self._copy()
        return new.slurm_(**kwargs)

    def slurm_(self, **kwargs):
        """
        In-place version of `Launcher.slurm()`.
        """
        for key, value in kwargs.items():
            if not hasattr(self._slurm, key):
                raise AttributeError(f"Invalid Slurm config {key}")
            setattr(self._slurm, key, value)
        return self

    def __call__(self, *args, **kwargs):
        """
        Schedule an XP with the current default training hyper-parameters
        and Slurm config. You can also provide extra overrides like in `bind()`.
        """
        launcher = self.bind(*args, **kwargs)
        if self._pool is None:
            sheep = self._shepherd.get_sheep_from_argv(launcher._argv)
            self._herd[sheep.xp.sig] = (sheep, launcher._slurm)
        else:
            future = self._pool.submit(_process, launcher._shepherd,
                                       launcher._argv, launcher._slurm)
            self._herd[len(self._herd)] = future


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
            tt.leaf("sig", align=">"),
            tt.leaf("sid", align=">"),
        ]

    def get_colors(self):
        return ["0", "38;5;245"]

    def process_history(self, history: tp.List[dict]) -> dict:
        """Process history to return a dict (with possibly nested dict inside)
        matching the schema given by `get_grid_metrics`.
        """
        out = {
            'epoch': len(history)
        }
        for metrics in history:
            out.update(metrics)
        return out
