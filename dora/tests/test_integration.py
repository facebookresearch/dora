import os
import pickle
import subprocess as sp

import pytest


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

    with pytest.raises(sp.SubprocessError):
        run_cmd(["run", "--git_save"])

    os.environ['_DORA_CLEAN_GIT'] = '1'
    try:
        with pytest.raises(sp.SubprocessError):
            # this one will fail because of the internal check in test_main.py
            run_cmd(["run"])
        run_cmd(["run", '--git_save'])
        # Testing a second time, to make sure updating an existing repo works fine.
        run_cmd(["run", '--git_save'])
    finally:
        os.environ['_DORA_CLEAN_GIT'] = '0'


def test_pickle(tmpdir):
    os.environ['_DORA_TEST_TMPDIR'] = str(tmpdir)
    from .integ.train import main

    main._full_name = "dora.tests.integ.train.main"

    other = pickle.loads(pickle.dumps(main))
    assert other is main
