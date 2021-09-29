# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import torch

import pytest

from ..conf import DoraConfig
from ..xp import XP


class _Cfg:
    pass


def get_dora(tmpdir: Path):
    return DoraConfig(dir=Path(tmpdir), exclude=["a"])


def test_dora_dir_abs():
    dora = get_dora('outputs')
    assert dora.dir.is_absolute()
    dora.dir = 'plop'
    assert dora.dir.is_absolute()


def test_sig(tmpdir):
    tmpdir = Path(str(tmpdir))
    dora = get_dora(tmpdir)
    xp = XP(dora=dora, cfg=_Cfg(), argv=[], delta=[("a", 5), ("b", 12)])
    assert xp.sig is not None

    xp2 = XP(dora=dora, cfg=_Cfg(), argv=[], delta=[("a", 12), ("b", 12)])
    assert xp.sig == xp2.sig

    xp3 = XP(dora=dora, cfg=_Cfg(), argv=[], delta=[("a", 12), ("b", 24)])
    assert xp.sig != xp3.sig


def test_properties(tmpdir):
    tmpdir = Path(str(tmpdir))
    dora = get_dora(tmpdir)

    xp = XP(dora=dora, cfg=_Cfg(), argv=[], delta=[("a", 5), ("b", 12)])
    xp.folder.relative_to(tmpdir)
    xp.submitit.relative_to(tmpdir)
    xp.rendezvous_file.relative_to(tmpdir)
    xp.history.relative_to(tmpdir)
    xp._argv_cache.relative_to(tmpdir)


def test_link(tmpdir):
    tmpdir = Path(str(tmpdir))
    dora = get_dora(tmpdir)
    xp = XP(dora=dora, cfg=_Cfg(), argv=[], delta=[("a", 5), ("b", 12)])
    xp.folder.mkdir(parents=True)

    xp.link.push_metrics({"plop": 42})

    xp = XP(dora=dora, cfg=_Cfg(), argv=[], delta=[("a", 5), ("b", 12)])
    assert xp.link.history == []
    xp.link.load()
    assert xp.link.history == [{"plop": 42}]

    val = [{"plok": 43, "out": Path("plop"), "mat": torch.zeros(5)}]
    xp.link.update_history(val)
    assert xp.link.history == [{"plok": 43, "out": "plop", "mat": [0.] * 5}]
    with pytest.raises(ValueError):
        xp.link.update_history({"plop": 42})

    with pytest.raises(ValueError):
        xp.link.update_history([{"plop": object()}])
