from pathlib import Path
import sys

import pytest

from ..hydra import hydra_main
from ..git_save import git_save, to_absolute_path
from ..xp import get_xp, XP

_ret = None

current_path = Path('.').resolve()


def _main(cfg):
    global _ret
    xp = get_xp()
    xp.link.push_metrics({"loss": 0.1})
    _ret = xp  # hydra does not support return values
    assert to_absolute_path('.') == str(current_path), (to_absolute_path('.'), current_path)


def get_main(tmpdir):
    tmpdir = Path(str(tmpdir))
    dora_main = hydra_main(config_path="./test_conf", config_name="test_conf")(_main)
    dora_main.dora.dir = tmpdir
    return dora_main


def call(main, argv):
    old_argv = list(sys.argv)
    try:
        sys.argv[1:] = argv
        main()
    finally:
        sys.argv = old_argv
    return _ret


def test_hydra_git_save(tmpdir):
    _main.__module__ = __name__
    main = get_main(tmpdir)
    argv = ['optim.loss=git_save']
    xp = main.get_xp(argv)

    with git_save(xp, True):
        call(main, argv)


def test_hydra(tmpdir):
    _main.__module__ = __name__
    main = get_main(tmpdir)
    xp = call(main, [])
    assert isinstance(xp, XP)
    assert len(xp.sig) > 0

    assert main.get_slurm_config().cpus_per_task == 5

    argv = ["num_workers=40"]
    xp2 = call(main, argv)
    assert xp.sig == xp2.sig
    assert xp2.cfg.num_workers == 40

    argv = main.value_to_argv({"useless.a": 3})
    assert len(argv) > 0
    xp2 = call(main, argv)
    assert xp.sig == xp2.sig
    assert xp2.cfg.useless.a == 3

    pre = ["useless.b=false", "optim.loss=l1"]
    argv = main.value_to_argv(pre)
    assert argv == pre
    xp2 = call(main, pre)
    assert xp.sig != xp2.sig

    assert argv == main.get_argv_from_sig(xp2.sig)

    xp3 = main.get_xp_from_sig(xp2.sig)
    assert xp2.argv == xp3.argv
    assert xp2.delta == xp3.delta
    assert xp2.sig == xp3.sig
    assert xp2.dora == xp3.dora

    metrics = main.get_xp_history(xp3)
    assert metrics[-1]["loss"] == 0.1

    name = main.get_name(xp3)
    assert name == "opt.loss=l1"

    argv = ["+k=youpi"]
    with pytest.raises(AssertionError):
        xp2 = call(main, argv)

    with pytest.raises(ValueError):
        main.value_to_argv(0.5)

    argv = ["plop.b=5"]
    xp2 = call(main, argv)
    assert xp2.cfg.plop.b == 5
    assert not hasattr(xp2.cfg, 'lapin')

    argv = ["group=lapin"]
    xp2 = call(main, argv)
    assert xp2.cfg.lapin.a == 5
    assert not hasattr(xp2.cfg, 'plop')

    argv = ["group=lapin", "plop.b=5"]
    with pytest.raises(Exception):
        xp2 = call(main, argv)
