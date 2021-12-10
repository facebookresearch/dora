# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from pathlib import Path
import os
import sys

import pytest

from ..main import argparse_main
from ..xp import XP, get_xp


parser = ArgumentParser("test_main")
parser.add_argument("--a", type=int)
parser.add_argument("--b", type=int, default=42)
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--cat_a")
parser.add_argument("--cat_b", type=int)


EXCLUDE = ["num_workers", "cat_*"]


def get_main(dora_dir, shared=None):
    @argparse_main(
        parser=parser, exclude=EXCLUDE, dir=dora_dir,
        shared=shared, use_underscore=True)
    def main():
        xp = get_xp()
        cwd = str(Path('.').resolve())
        code = str((xp.folder / 'code').resolve())
        if xp.dora.git_save:
            assert cwd.startswith(code), cwd
            assert __file__.startswith(code), __file__
        else:
            assert not cwd.startswith(code), cwd
            assert not __file__.startswith(code), __file__
        xp.link.push_metrics({"loss": 0.1})
        return xp

    if os.environ.get('_DORA_GIT_SAVE') == '1':
        main.dora.git_save = True
    return main


def call(main, argv):
    old_argv = list(sys.argv)
    try:
        sys.argv[1:] = argv
        return main()
    finally:
        sys.argv = old_argv


def test_shared(tmpdir):
    shared = tmpdir / 'shared'
    main_a = get_main(tmpdir / 'a', shared)
    main_b = get_main(tmpdir / 'b')
    main_c = get_main(tmpdir / 'c', shared)

    xp = main_a.get_xp(['--a=5'])
    main_a.init_xp(xp)

    with pytest.raises(RuntimeError):
        main_b.get_xp_from_sig(xp.sig)
    xp2 = main_c.get_xp_from_sig(xp.sig)
    assert xp2.sig == xp.sig
    assert xp2.argv == xp.argv


def test_main(tmpdir):
    main = get_main(tmpdir)
    xp = call(main, [])
    assert isinstance(xp, XP)
    assert len(xp.sig) > 0

    argv = ["--cat_a=plop"]
    xp2 = call(main, argv)
    assert xp.sig == xp2.sig

    argv = main.value_to_argv({"b": 42, "cat_b": 5})
    assert len(argv) > 0
    xp2 = call(main, argv)
    assert xp.sig == xp2.sig
    assert xp2.cfg.cat_b == 5

    pre = ["--cat_a=plop", "--a=4"]
    argv = main.value_to_argv(pre)
    assert argv == pre
    xp2 = call(main, pre)
    assert xp.sig != xp2.sig

    assert argv == main.get_argv_from_sig(xp2.sig)

    xp3 = main.get_xp_from_sig(xp2.sig)
    assert xp2 == xp3

    metrics = main.get_xp_history(xp3)
    assert metrics[-1]["loss"] == 0.1

    name = main.get_name(xp3)
    assert name == "a=4"

    with pytest.raises(ValueError):
        main.value_to_argv(0.5)
