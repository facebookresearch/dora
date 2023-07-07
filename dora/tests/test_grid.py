# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ..conf import SubmitRules
from ..explore import Explorer, Launcher
from ..hydra import HydraMain
from ..grid import run_grid, RunGridArgs
from .fake_shep import mock_shep
from .test_main import get_main
from .test_hydra import get_main as get_main_hydra

_ret = None


def explore_1(launcher: Launcher):
    launcher()
    launcher(num_workers=42)


def explore_2(launcher: Launcher):
    launcher(num_workers=42, a=4)


def test_shep(tmpdir):
    def rgrid(explore):
        return run_grid(main, Explorer(explore), "unittest",
                        slurm=slurm, rules=rules, args=args)
    with mock_shep():
        main = get_main(tmpdir)
        slurm = main.get_slurm_config()
        rules = SubmitRules()
        args = RunGridArgs()

        args.monitor = False
        args.dry_run = True

        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].job is None

        args.dry_run = False
        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].job.job_id == "0"
        assert not sheeps[0].is_done()

        args.cancel = True
        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].state() == "CANCELLED"
        assert sheeps[0].is_done()

        args.cancel = False
        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].state() == "CANCELLED"

        rules.retry = True
        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].state() == "UNKNOWN"
        assert sheeps[0].job.job_id == "1"

        old_sheep = sheeps[0]

        args.verbose = True
        sheeps = rgrid(explore_2)
        assert len(sheeps) == 1
        assert sheeps[0].state() == "UNKNOWN"
        assert sheeps[0].job.job_id == "2"
        assert old_sheep.state() == "CANCELLED"


def explore_hydra(launcher: Launcher):
    launcher.bind_({'epochs': 50, 'optim.loss': '123', 'num_workers': None})
    launcher({'complex.a': [{"test": "weird"}]})
    launcher({'complex.b': {"a": 21, "b": 4}})
    launcher({'+complex.b': {"a": 21, "b": 4, "c": 13}})


def test_shep_hydra(tmpdir):
    def rgrid(explore):
        return run_grid(main, Explorer(explore), "unittest",
                        rules=rules, args=args)
    HydraMain._slow = False
    with mock_shep():
        main = get_main_hydra(tmpdir)
        rules = SubmitRules()
        args = RunGridArgs()
        args.monitor = False
        args.dry_run = True

        sheeps = rgrid(explore_hydra)
        assert len(sheeps) == 3
        cfg = sheeps[0].xp.cfg
        assert cfg.epochs == 50
        assert cfg.optim.loss == '123'
        assert cfg.num_workers is None
        assert cfg.complex.a == [{"test": "weird"}]

        cfg = sheeps[1].xp.cfg
        assert cfg.complex.b == {"a": 21, "b": 4}

        cfg = sheeps[2].xp.cfg
        assert cfg.complex.b == {"a": 21, "b": 4, "c": 13}
