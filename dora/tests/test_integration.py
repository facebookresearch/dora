import os
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
