# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Classes used to define a grid search.

`Launcher`: a launcher is passed to each grid search explore function,
and can be called repeatidly to schedule XPs.

`Explorer`: defines some metadata, in particular the metrics to display
with the `dora grid` command.
"""
from collections import OrderedDict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
import typing as tp

from treetable.table import _Node
import treetable as tt

from .conf import SlurmConfig
from .shep import Shepherd, Sheep


class ProcessException(RuntimeError):
    pass


def _process(shepherd: Shepherd, argv: tp.List[str], slurm: SlurmConfig,
             job_array_index: tp.Optional[int] = None):
    try:
        return (shepherd.get_sheep_from_argv(argv), slurm, job_array_index)
    except Exception as exc:
        raise ProcessException(repr(exc))


@dataclass
class Herd:
    """Represents a herd of sheeps ready to be scheduled.
    """
    sheeps: tp.Dict[str, Sheep] = field(default_factory=OrderedDict)
    slurm_configs: tp.Dict[str, SlurmConfig] = field(default_factory=dict)
    job_arrays: tp.List[tp.List[str]] = field(default_factory=list)

    # Sheeps that need to be evaluated in a process pool for faster execution.
    _pendings: tp.List[Future] = field(default_factory=list)

    _job_array_launcher: tp.Optional["Launcher"] = None

    def complete(self):
        """Complete all pending sheep evaluations and add them to the herd."""
        while self._pendings:
            future = self._pendings.pop(0)
            sheep, slurm, job_array_index = future.result()
            self._add_sheep(sheep, slurm, job_array_index)

    def add_sheep(self, shepherd: Shepherd, argv: tp.List[str], slurm: SlurmConfig,
                  pool: tp.Optional[ProcessPoolExecutor] = None):
        if self._job_array_launcher is None:
            self.job_arrays.append([])
        job_array_index = len(self.job_arrays) - 1
        if pool is None:
            self._add_sheep(shepherd.get_sheep_from_argv(argv), slurm, job_array_index)
        else:
            self._pendings.append(pool.submit(_process, shepherd, argv, slurm, job_array_index))

    def _add_sheep(self, sheep: Sheep, slurm: SlurmConfig,
                   job_array_index: tp.Optional[int] = None):
        if sheep.xp.sig in self.sheeps:
            return
        self.sheeps[sheep.xp.sig] = sheep
        self.slurm_configs[sheep.xp.sig] = slurm
        if job_array_index is not None:
            self.job_arrays[job_array_index].append(sheep.xp.sig)


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

    def __init__(self, shepherd: Shepherd, slurm: SlurmConfig, herd: Herd,
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
        array_launcher = self._herd._job_array_launcher
        if array_launcher is not None:
            assert array_launcher._slurm == launcher._slurm, \
                "cannot change slurm config inside job array."
        self._herd.add_sheep(self._shepherd, launcher._argv, launcher._slurm, self._pool)

    @contextmanager
    def job_array(self):
        """Context manager to indicate that you wish to launch all the included
        XPs using a single job array with the current Slurm parameters.
        """
        assert self._herd._job_array_launcher is None, "Cannot stack job arrays"
        self._herd._job_array_launcher = self._copy()
        self._herd.job_arrays.append([])
        try:
            yield
        finally:
            self._herd._job_array_launcher = None


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

    def process_sheep(self, sheep: Sheep, history: tp.List[dict]) -> dict:
        """Process a sheep to return a dict (with possibly nested dict inside)
        matching the schema given by `get_grid_metrics`.
        This gives more possiblities than `process_history`, which is kept for compatibility,
        as one has access to the XP config here.
        If this is implemented, it will always be called, otherwise, `process_history` is used.

        One should use the history provided here, rather than the one in `sheep.xp.link.history`,
        as it has possibly been shortened to align multiple experiments.
        """
        raise NotImplementedError()

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
