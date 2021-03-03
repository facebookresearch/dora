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
from hashlib import sha1
import json
from pathlib import Path
import typing as tp
import sys

from .conf import DoraConfig, XP
from .customize import custom, Customizations
from .utils import jsonable


def _get_sig(jsoned: tp.Any) -> str:
    return sha1(json.dumps(jsoned).encode('utf8')).hexdigest()[:8]


class _Context:
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


def get_run() -> XP:
    if _context._run is None:
        raise RuntimeError("Not in a run!")
    else:
        return _context._run


MainFun = tp.Callable[[], tp.Any]


class DecoratedMain:
    def __init__(self, main: MainFun, dora: DoraConfig, custom: Customizations = custom):
        self.main = main
        self.dora = dora
        self.custom = custom

    def __call__(self):
        argv = self._get_argv()
        run = self.get_run(argv)
        self._init_run(run)
        with _context.enter_run(run):
            self.main()

    def get_run(self, argv: tp.Sequence[str]) -> XP:
        raise NotImplementedError()

    def _get_argv(self) -> tp.List[str]:
        return sys.argv[1:]

    def _init_run(self, run: XP):
        run.folder.mkdir(exist_ok=True, parents=True)
        json.dump(run.argv, open(run._argv_cache, 'w'))
        return run

    def get_argv_from_sig(self, sig: str) -> tp.Sequence[str]:
        run = XP(sig=sig, dora=self.dora, cfg=None, argv=[])
        if run._argv_cache.exists():
            return json.load(open(run._argv_cache))
        else:
            raise RuntimeError(f"Could not find experiment with signature {sig}")

    def get_run_from_sig(self, sig: str) -> XP:
        return self.get_run(self.get_argv_from_sig)

    def __repr__(self):
        return f"DecoratedMain({self.main})"

    def merge_args(self, args: tp.List[tp.Any]) -> tp.List[str]:
        raise NotImplementedError()

    def get_name_parts(self, run: XP) -> OrderedDict:
        raise NotImplementedError()

    def get_name(self, run: XP) -> str:
        short_parts = [self.custom.short_name_part(k, v)
                       for k, v in self.get_name_parts(run).items()]
        short_parts.insert(0, run.dora.name)
        return " ".join(short_parts)


class ArgparseMain(DecoratedMain):
    def __init__(self, main: XP, dora: DoraConfig, parser: argparse.ArgumentParser,
                 custom: Customizations = custom):
        super().__init__(main, dora, custom)
        self.parser = parser

    def get_run(self, argv: tp.Sequence[str]) -> XP:
        argv = list(argv)
        args = self.parser.parse_args(argv)
        delta = []
        for key, value in args.__dict__.items():
            if self.parser.get_default(key) != value:
                if not self.dora.is_excluded(key):
                    delta.append((key, value))
        delta_sorted = sorted(delta, key=lambda x: x[0])
        sig = _get_sig(jsonable(delta_sorted))
        run = XP(sig=sig, dora=self.dora, cfg=args, argv=argv, delta=delta)
        return run

    def merge_args(self, args: tp.List[tp.Any]) -> tp.List[str]:
        argv = []
        for part in args:
            if not isinstance(part, list):
                raise ValueError("Can only merge list of argv.")
            argv += part
        return argv

    def get_name_parts(self, run: XP) -> OrderedDict:
        parts = OrderedDict()
        for name, value in run.delta:
            parts[name] = value
        return parts


def argparse_main(parser, name: str = None,
                  exclude: tp.Sequence[str] = None,
                  dir: tp.Union[str, Path] = "./outputs",
                  custom: Customizations = custom):
    def _decorator(main):
        dora = DoraConfig(
            name=name or main.__module__,
            dir=Path(dir),
            exclude=exclude or [])
        return ArgparseMain(main, dora, parser, custom)
    return _decorator
