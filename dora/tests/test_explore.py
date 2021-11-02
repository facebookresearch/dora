# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from .test_main import get_main
from ..explore import Launcher, Explorer, Herd
from ..shep import Shepherd


def test_launcher(tmpdir):
    def get_xp(idx):
        return list(herd.sheeps.values())[idx].xp

    def get_slurm(idx):
        return herd.slurm_configs[get_xp(idx).sig]

    main = get_main(tmpdir)
    slurm = main.get_slurm_config()
    shepherd = Shepherd(main)
    herd = Herd()

    launcher = Launcher(shepherd, slurm, herd)

    def explore(launcher):
        launcher(num_workers=20)
        assert len(herd.sheeps) == 1
        assert len(herd.job_arrays) == 1
        assert get_xp(0).cfg.num_workers == 20

        launcher(num_workers=40)
        assert len(herd.sheeps) == 1
        assert len(herd.job_arrays) == 2
        assert herd.job_arrays[1] == []

        launcher.bind_(a=5)
        launcher()
        assert len(herd.sheeps) == 2
        assert len(herd.job_arrays) == 3
        assert herd.job_arrays[2] == [get_xp(1).sig]
        assert get_xp(1).cfg.a == 5

        sub = launcher.bind({"num_workers": 150}, a=6)
        launcher()
        assert len(herd.sheeps) == 2

        sub.slurm_(cpus_per_task=40)
        sub()
        assert len(herd.sheeps) == 3
        assert get_xp(2).cfg.a == 6
        assert get_xp(2).cfg.num_workers == 150
        assert get_slurm(2).cpus_per_task == 40

        with pytest.raises(AttributeError):
            sub.slurm(cpu_per_task=40)

        before = len(herd.job_arrays)
        before_sheeps = len(herd.sheeps)
        with sub.job_array():
            for k in range(100):
                sub(a=100 + k)
            with pytest.raises(AssertionError):
                sub.slurm_(gpus=8)(a=1000)
        assert len(herd.job_arrays) == before + 1
        assert len(herd.sheeps) == before_sheeps + 100

    Explorer(explore)(launcher)
