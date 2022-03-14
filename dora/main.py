# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DecoratedMain is the main abstraction used inside Dora.
This defines the core interface that needs to be fullfilled
for Dora to be able to handle an experiment.

This is also where the end user will specify how to launch job, default
Slurm configuration, storage location, naming conventions etc.
"""

import argparse
from collections import OrderedDict
import importlib
import json
from pathlib import Path
import typing as tp
import sys

from .conf import DoraConfig, SlurmConfig
from .names import NamesMixin
from .xp import XP, _context


MainFun = tp.Callable


def _load_main(full_name):
    module_name, fun_name = full_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fun_name)


def get_module_name(module):
    if module == "__main__":
        spec = sys.modules[module].__spec__
        if spec is None:
            return None
        else:
            return spec.name
    else:
        return module


class DecoratedMain(NamesMixin):
    """
    Main function that will actually run the XP, wrapped with tons of meta-informations.

    Args:
        main (callable, taking no argument): main function for the experiment as provided
            by the user.
        dora (DoraConfig): configuration for Dora.
    """
    _slow = False

    def __init__(self, main: MainFun, dora: DoraConfig):
        self.main = main
        self.dora = dora
        module_name = get_module_name(main.__module__)
        if module_name is None:
            # we are being called in a weird way and definitely not from
            # a Dora command.
            self.package = 'unknown'
            self.main_module = 'train'
        else:
            if '.' in module_name:
                self.package, self.main_module = module_name.rsplit(".", 1)
            else:
                self.package = 'unknown'
                self.main_module = module_name

        self.name = self.package
        self._full_name = main.__module__ + "." + main.__name__

    def __call__(self):
        argv = self._get_argv()
        if not self._is_active(argv):
            return self._main()
        xp = self.get_xp(argv)
        self.init_xp(xp)
        with _context.enter_xp(xp):
            return self._main()

    def _is_active(self, argv: tp.List[str]) -> bool:
        return True

    def _main(self):
        return self.main()

    def __reduce__(self):
        return _load_main, (self._full_name,)

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
        """
        Initialize the XP folder. Once this is done, the XP can be retrieved
        from its signature. This is done automatically before running,
        or when using the `--init` flag of the `dora grid` command.

        This will also initialize the shared XP folder so that the XP hyper-params
        can be easily shared using its signature.
        """
        xp.folder.mkdir(exist_ok=True, parents=True)
        json.dump(xp.argv, open(xp._argv_cache, 'w'))
        if xp._shared_argv_cache is not None:
            # Create xps and XP folders with 0777 mode.
            xp._shared_argv_cache.parent.parent.mkdir(exist_ok=True, parents=True, mode=0o777)
            xp._shared_argv_cache.parent.mkdir(exist_ok=True, parents=True, mode=0o777)
            try:
                xp._shared_argv_cache.parent.parent.chmod(0o777)
                xp._shared_argv_cache.parent.chmod(0o777)
            except PermissionError:
                pass
            json.dump(xp.argv, open(xp._shared_argv_cache, 'w'))
            try:
                xp._shared_argv_cache.chmod(0o777)
            except PermissionError:
                pass
        return xp

    def get_argv_from_sig(self, sig: str) -> tp.Sequence[str]:
        """Returns the argv used to obtain a given signature.
        This can only work if an XP was previously ran with this signature.
        """
        xp = XP(sig=sig, dora=self.dora, cfg=None, argv=[])
        if xp._argv_cache.exists():
            return json.load(open(xp._argv_cache))
        elif xp._shared_argv_cache is not None and xp._shared_argv_cache.exists():
            return json.load(open(xp._shared_argv_cache))
        else:
            raise RuntimeError(f"Could not find experiment with signature {sig}")

    def get_xp_from_sig(self, sig: str) -> XP:
        """Returns the XP from the signature. Can only work if such an XP
        has previously ran.
        """
        return self.get_xp(self.get_argv_from_sig(sig))

    def __repr__(self):
        return f"DecoratedMain({self.main})"

    def value_to_argv(self, arg: tp.Any) -> tp.List[str]:
        """Convert a Python value to argv. arg can be either:
        - a list, then each entry will be converted and all argv are concatenated.
        - a str, then it is directly an argv entry.
        - a dict, then each key, value pair is mapped to an argv entry.
        """
        raise NotImplementedError()

    def get_xp_history(self, xp: XP) -> tp.List[dict]:
        """Return the metrics for a given XP. By default this will look into
        the `history.json` file, that can be populated with the Link class.

        Can be overriden, but metrics should still be returned as a list
        of dicts, possibly with nested dicts.
        """
        xp.link.load()
        return xp.link.history

    def get_slurm_config(self) -> SlurmConfig:
        """Return default Slurm config for the launch and grid actions.
        """
        return SlurmConfig()


class ArgparseMain(DecoratedMain):
    """Implementation of `DecoratedMain` for XP that uses argparse.

    Args:
        main : main function to wrap.
        dora : Dora config, containing the exclude and dir fields.
        parser : parser to use, and to derive default values from.
        slurm : default slurm config for scheduling jobs.
        use_underscore : if False, scheduling a job as `launcher(batch_size=32)`
            will translate to the command-line `--batch-size=32`,
            otherwise, it will stay as `--batch_size=32`.
    """
    def __init__(self, main: MainFun, dora: DoraConfig, parser: argparse.ArgumentParser,
                 slurm: tp.Optional[SlurmConfig] = None, use_underscore: bool = True):
        super().__init__(main, dora)
        self.parser = parser
        self.use_underscore = use_underscore
        self.slurm = slurm

    def get_xp(self, argv: tp.Sequence[str]) -> XP:
        argv = list(argv)
        args = self.parser.parse_args(argv)
        delta = []
        for key, value in args.__dict__.items():
            if self.parser.get_default(key) != value:
                delta.append((key, value))
        xp = XP(dora=self.dora, cfg=args, argv=argv, delta=delta)
        return xp

    def value_to_argv(self, arg: tp.Any) -> tp.List[str]:
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
        elif isinstance(arg, (list, tuple)):
            for part in arg:
                argv += self.value_to_argv(part)
        else:
            raise ValueError(f"Can only process dict, tuple, lists and str, but got {arg}")
        return argv

    def get_name_parts(self, xp: XP) -> OrderedDict:
        parts = OrderedDict()
        assert xp.delta is not None
        for name, value in xp.delta:
            parts[name] = value
        return parts

    def get_slurm_config(self) -> SlurmConfig:
        """Return default Slurm config for the launch and grid actions.
        """
        if self.slurm is not None:
            return self.slurm
        return super().get_slurm_config()


def argparse_main(parser: argparse.ArgumentParser, *,
                  dir: tp.Union[str, Path] = "./outputs",
                  exclude: tp.Sequence[str] = [],
                  slurm: tp.Optional[SlurmConfig] = None,
                  shared: tp.Optional[tp.Union[str, Path]] = None,
                  use_underscore: bool = True,
                  **kwargs):
    """Nicer version of `ArgparseMain` that acts like a decorator, and directly
    exposes the most useful configs to override.

    Args:
        parser: parser to use, and to derive default values from.
        exclude: list of patterns of arguments to exclude from the computation
            of the XP signature.
        dir: path to store logs, checkpoints, etc. to.
        slurm: default slurm config for scheduling jobs.
        shared: path to the shared XP repository.
        use_underscore: if False, scheduling a job as `launcher(batch_size=32)`
            will translate to the command-line `--batch-size=32`,
            otherwise, it will stay as `--batch_size=32`.
        **kwargs: extra args are passed to `DoraConfig`.
    """
    def _decorator(main: MainFun):
        dora = DoraConfig(
            dir=Path(dir),
            shared=None if shared is None else Path(shared),
            exclude=list(exclude),
            **kwargs)
        return ArgparseMain(main, dora, parser, use_underscore=use_underscore, slurm=slurm)
    return _decorator
