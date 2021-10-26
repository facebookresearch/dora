# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ..conf import SubmitRules
from ..shep import Shepherd, _JobArray
from .fake_shep import mock_shep
from .test_main import get_main

_ret = None


def test_shep(tmpdir):
    with mock_shep():
        main = get_main(tmpdir)
        shepherd = Shepherd(main)
        slurm = main.get_slurm_config()
        rules = SubmitRules()

        sheep = shepherd.get_sheep_from_argv([])
        assert sheep.job is None
        shepherd._submit(_JobArray(slurm, [sheep]))
        assert sheep.job is not None
        assert sheep.job.job_id == "0"

        shepherd.maybe_submit_lazy(sheep, slurm, rules)
        shepherd.commit()
        assert sheep.job.job_id == "0"
        old_job = sheep.job

        rules.replace = True
        shepherd.maybe_submit_lazy(sheep, slurm, rules)
        shepherd.commit()
        assert sheep.job.job_id == "1"
        assert old_job._state == "CANCELLED"

        sheep2 = shepherd.get_sheep_from_job_id("1")
        assert sheep.xp == sheep2.xp
        assert sheep.job.job_id == "1"

        sheep2 = shepherd.get_sheep_from_job_id("0")
        assert sheep.xp == sheep2.xp

        sheep2 = shepherd.get_sheep_from_job_id("2")
        assert sheep2 is None

        sheep.job._state = "FAILED"
        rules.replace = False
        shepherd.maybe_submit_lazy(sheep, slurm, rules)
        assert sheep.job.job_id == "1"
        rules.retry = True
        shepherd.maybe_submit_lazy(sheep, slurm, rules)
        shepherd.commit()
        assert sheep.job.job_id == "2"

        sheep.job._state = "COMPLETED"
        rules.replace = True
        rules.retry = True
        shepherd.maybe_submit_lazy(sheep, slurm, rules)
        shepherd.commit()
        assert sheep.job.job_id == "2"

        rules.replace_done = True
        shepherd.maybe_submit_lazy(sheep, slurm, rules)
        shepherd.commit()
        assert sheep.job.job_id == "3"

        main.dora.git_save = True
        sheep = shepherd.get_sheep_from_argv(["--a=56"])
        shepherd.maybe_submit_lazy(sheep, slurm, rules)
        shepherd.commit()
        assert sheep.xp.code_folder.name == 'code'
        assert sheep.xp.code_folder.exists()
