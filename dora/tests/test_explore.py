from collections import OrderedDict

import pytest

from .test_main import get_main
from ..explore import Launcher, Explorer
from ..shep import Shepherd


def test_launcher(tmpdir):
    def get_xp(idx):
        return list(herd.values())[idx][0].xp

    def get_slurm(idx):
        return list(herd.values())[idx][1]

    main = get_main(tmpdir)
    slurm = main.get_slurm_config()
    shepherd = Shepherd(main)
    herd = OrderedDict()

    launcher = Launcher(shepherd, slurm, herd)

    def explore(launcher):
        launcher(num_workers=20)
        assert len(herd) == 1
        assert get_xp(0).cfg.num_workers == 20

        launcher(num_workers=40)
        assert len(herd) == 1
        assert get_xp(0).cfg.num_workers == 40

        launcher.bind_(a=5)
        launcher()
        assert len(herd) == 2
        assert get_xp(1).cfg.a == 5

        sub = launcher.bind({"num_workers": 150}, a=6)
        launcher()
        assert len(herd) == 2

        sub.slurm_(cpus_per_task=40)
        sub()
        assert len(herd) == 3
        assert get_xp(2).cfg.a == 6
        assert get_xp(2).cfg.num_workers == 150
        assert get_slurm(2).cpus_per_task == 40

        with pytest.raises(AttributeError):
            sub.slurm(cpu_per_task=40)

    Explorer(explore)(launcher)
