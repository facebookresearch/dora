"""
DecoratedMain is the main abstraction used inside Dora.
This defines the core interface that needs to be fullfilled
for Dora to be able to handle an experiment.
"""

import argparse
from contextlib import contextmanager
from hashlib import sha1
import json
import typing as tp
import sys

from .conf import DoraConfig, DoraRun
from .utils import jsonable


def _get_sig(jsoned) -> str:
    return sha1(json.dumps(jsoned).encode('utf8')).hexdigest()[:8]


class _Context:
    def __init__(self):
        self._run: DoraRun = None

    @contextmanager
    def enter_run(self, run: DoraRun):
        if self._run is not None:
            raise RuntimeError("Already in a run.")
        self._run = run
        try:
            yield
        finally:
            self._run = None


_context = _Context()


def get_run() -> DoraRun:
    if _context._run is None:
        raise RuntimeError("Not in a run!")
    else:
        return _context._run


class DecoratedMain:
    def __init__(self, main, dora: DoraConfig):
        self.main = main
        self.dora = dora

    def __call__(self):
        argv = self._get_argv()
        run = self.get_run(argv)
        with _context.enter_run(run):
            self.main()

    def get_run(self, argv: tp.Sequence[str]) -> DoraRun:
        raise NotImplementedError()

    def _get_argv(self) -> tp.List[str]:
        return sys.argv[1:]

    def _init_run(self, run):
        run.folder.mkdir(exist_ok=True, parents=True)
        json.dump(run.argv, open(run._argv_cache, 'w'))
        return run

    def get_argv_from_sig(self, sig: str):
        run = DoraRun(sig=sig, dora=self.dora, cfg=None, argv=[])
        if run._argv_cache.exists():
            return json.load(open(run._argv_cache))
        else:
            raise RuntimeError(f"Could not find experiment with signature {sig}")

    def get_run_from_sig(self, sig: str):
        return self.get_run(self.get_argv_from_sig)

    def __repr__(self):
        return f"DecoratedMain({self.main})"


class ArgparseMain(DecoratedMain):
    def __init__(self, main, dora: DoraConfig, parser: argparse.ArgumentParser):
        super().__init__(main, dora)
        self.parser = parser

    def get_run(self, argv: tp.Sequence[str]) -> DoraRun:
        argv = list(argv)
        args = self.parser.parse_args(argv)
        delta = []
        for key, value in args.__dict__.items():
            if self.parser.get_default(key) != value:
                if not self.dora.is_excluded(key):
                    delta.append((key, value))
        delta.sort(key=lambda x: x[0])
        sig = _get_sig(jsonable(delta))
        run = DoraRun(sig=sig, dora=self.dora, cfg=args, argv=argv)
        self._init_run(run)
        return run


def argparse_main(parser, name=None, exclude=None, dir="./outputs"):
    def _decorator(main):
        dora = DoraConfig(
            name=name or main.__module__,
            dir=dir,
            exclude=exclude or [])
        return ArgparseMain(main, dora, parser)
    return _decorator
