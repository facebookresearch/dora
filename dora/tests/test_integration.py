import os
import pickle
import shutil
import subprocess as sp

import pytest

from .test_main import get_main


def run_cmd(argv):
    return sp.run(["dora", "-P", "dora.tests.integ"] + argv, check=True, capture_output=False)


def test_integration(tmpdir):
    os.environ['_DORA_TEST_TMPDIR'] = str(tmpdir)
    with pytest.raises(sp.SubprocessError):
        run_cmd(["info", "--", "a=32"])
    run_cmd(["info"])
    run_cmd(["run"])
    run_cmd(["grid", "test", "--dry_run", "--no_monitor"])
    run_cmd(["info", "--", "--a=32"])
    run_cmd(["--main_module", "other_train", "run"])


def test_git_save(tmpdir):
    os.environ['_DORA_TEST_TMPDIR'] = str(tmpdir)
    os.environ['_DORA_GIT_SAVE'] = '1'
    try:
        main = get_main(tmpdir)
        xp = main.get_xp([])
        code = xp.code_folder

        run_cmd(["run"])
        assert not code.exists()

        run_cmd(["run", '--git_save'])
        assert code.exists()
        # Testing a second time, to make sure updating an existing repo works fine.
        run_cmd(["run", '--git_save'])
    finally:
        os.environ['_DORA_GIT_SAVE'] = '0'

    shutil.rmtree(code)
    run_cmd(["run", '--git_save'])
    assert code.exists()


def test_pickle(tmpdir):
    os.environ['_DORA_TEST_TMPDIR'] = str(tmpdir)
    from .integ.train import main

    main._full_name = "dora.tests.integ.train.main"

    other = pickle.loads(pickle.dumps(main))
    assert other is main
