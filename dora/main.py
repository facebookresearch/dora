"""
DecoratedMain is the main abstraction used inside Dora.
This defines the core interface that needs to be fullfilled
for Dora to be able to handle an experiment.

This is also where the end user will specify how to launch job, default
Slurm configuration, storage location, naming conventions etc.
"""

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import json
from pathlib import Path
import typing as tp
import sys

from .conf import DoraConfig, SlurmConfig
from .names import _NamesMixin
from .xp import XP


class _Context:
    # Used to keep track of a running XP and be able to provide
    # it on demand with `get_xp`.
    def __init__(self):
        self._run: XP = None

    @contextmanager
    def enter_run(self, run: XP):
        if self._run is not None:
            raise RuntimeError("Already in a run.")
        self._run = run
        try:
            yield
        finally:
            self._run = None


_context = _Context()


def get_xp() -> XP:
    """When running from within an XP, returns the XP object.
    Otherwise, raises RuntimeError.
    """
    if _context._run is None:
        raise RuntimeError("Not in a run!")
    else:
        return _context._run


MainFun = tp.Callable


class DecoratedMain(_NamesMixin):
    """
    Main function that will actually run the XP, wrapped with tons of meta-informations.

    Args:
        main (callable, taking no argument): main function for the experiment as provided
            by the user.
        dora (DoraConfig): configuration for Dora.
    """
    def __init__(self, main: MainFun, dora: DoraConfig):
        self.main = main
        self.dora = dora
        self.name = main.__module__.rsplit(".", 1)[0]

    def __call__(self):
        argv = self._get_argv()
        xp = self.get_xp(argv)
        self.init_xp(xp)
        with _context.enter_run(xp):
            return self._main()

    def _main(self):
        return self.main()

    def get_xp(self, argv: tp.Sequence[str]) -> XP:
        """Return an XP given a list of arguments.
        """
        raise NotImplementedError()

    def _get_argv(self) -> tp.List[str]:
        # Returns the actual list of arguments, typically from sys.argv.
        # This is only called when the XP is executed, not when it is obtained
        # by other means, e.g. from a grid search file, or info command.
        return sys.argv[1:]

    def init_xp(self, xp: XP):
        # Initialize the XP just before the actual execution.
        # This will create the XP directory and dump the argv used.
        xp.folder.mkdir(exist_ok=True, parents=True)
        json.dump(xp.argv, open(xp._argv_cache, 'w'))
        return xp

    def get_argv_from_sig(self, sig: str) -> tp.Sequence[str]:
        """Returns the argv used to obtain a given signature.
        This can only work if an XP was previously ran with this signature.
        """
        xp = XP(sig=sig, dora=self.dora, cfg=None, argv=[])
        if xp._argv_cache.exists():
            return json.load(open(xp._argv_cache))
        else:
            raise RuntimeError(f"Could not find experiment with signature {sig}")

    def get_xp_from_sig(self, sig: str) -> XP:
        """Returns the XP from the signature. Can only work if such an XP
        has previously ran.
        """
        return self.get_xp(self.get_argv_from_sig(sig))

    def __repr__(self):
        return f"DecoratedMain({self.main})"

    def merge_args(self, args: tp.List[tp.Any]) -> tp.List[str]:
        """Merge a sequence of arguments. This is used by the gird utility.
        """
        raise NotImplementedError()

    def get_xp_metrics(self, xp: XP) -> tp.List[dict]:
        """Return the metrics for a given XP. By default this will look into
        the `history.json` file, that can be populated with the Link class.

        Can be overriden, but metrics should still be returned as a list
        of dicts, possibly with nested dicts.
        """
        if xp.history.exists():
            metrics = json.load(open(xp.history))
            return metrics
        else:
            return []

    def get_slurm_config(self) -> SlurmConfig:
        """Return default Slurm config for the launch and grid actions.
        """
        return SlurmConfig()


class ArgparseMain(DecoratedMain):
    """Implementation of `DecoratedMain` for XP that uses argparse.
    """
    def __init__(self, main: MainFun, dora: DoraConfig, parser: argparse.ArgumentParser,
                 use_underscore=False):
        super().__init__(main, dora)
        self.parser = parser
        self.use_underscore = use_underscore

    def get_xp(self, argv: tp.Sequence[str]) -> XP:
        argv = list(argv)
        args = self.parser.parse_args(argv)
        delta = []
        for key, value in args.__dict__.items():
            if self.parser.get_default(key) != value:
                delta.append((key, value))
        xp = XP(dora=self.dora, cfg=args, argv=argv, delta=delta)
        return xp

    def grid_args_to_argv(self, arg: tp.Any) -> tp.List[str]:
        argv = []
        if isinstance(arg, str):
            argv.append(arg)
        elif isinstance(arg, dict):
            for key, value in arg.items():
                if not self.use_underscore:
                    key = key.replace("_", "-")
                if value is True:
                    argv.append(f"--{key}")
                else:
                    argv.append(f"--{key}={value}")
        elif isinstance(arg, [list, tuple]):
            for part in arg:
                argv += self.grid_args_to_argv(part)
        else:
            raise ValueError(f"Can only process dict, tuple, lists and str, but got {arg}")
        return argv

    def get_name_parts(self, xp: XP) -> OrderedDict:
        parts = OrderedDict()
        for name, value in xp.delta:
            parts[name] = value
        return parts


def argparse_main(parser: argparse.ArgumentParser, *, use_underscore: bool = False,
                  exclude: tp.Sequence[str] = None,
                  dir: tp.Union[str, Path] = "./outputs"):
    """Nicer version of `ArgparseMain` that acts like a decorator, and directly
    exposes the most useful configs to override.
    """
    def _decorator(main: MainFun):
        dora = DoraConfig(
            dir=Path(dir),
            exclude=exclude or [])
        return ArgparseMain(main, dora, parser, use_underscore=use_underscore)
    return _decorator
