# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Launch command.
"""
from functools import partial
import subprocess as sp
import time

from .conf import SubmitRules, update_from_args
from .main import DecoratedMain
from .shep import Shepherd
from .log import simple_log
from .utils import reliable_rmtree

log = partial(simple_log, "Launch:")


def launch_action(args, main: DecoratedMain):
    shepherd = Shepherd(main, log=log)
    slurm = main.get_slurm_config()
    update_from_args(slurm, args)
    rules = SubmitRules()
    update_from_args(rules, args)

    sheep = shepherd.get_sheep_from_argv(args.argv)
    log(f"Fetched sheep {sheep}")
    shepherd.update()
    if args.cancel:
        if sheep.job is None:
            log("Could not cancel non existing job")
        elif sheep.is_done():
            log("Job is not running")
        else:
            sheep.job.cancel()
        return

    if args.clear:
        log("Canceling current job...")
        if sheep.job is not None:
            shepherd.cancel_lazy(sheep.job)
        shepherd.commit()
        log("Deleting XP folder...")
        if sheep.xp.folder.exists():
            reliable_rmtree(sheep.xp.folder)
        sheep.job = None

    shepherd.maybe_submit_lazy(sheep, slurm, rules)
    shepherd.commit()

    if args.tail or args.attach:
        done = False
        tail_process = None
        wait = True
        try:
            while True:
                if sheep.log.exists() and tail_process is None:
                    tail_process = sp.Popen(["tail", "-n", "200", "-f", sheep.log])
                if sheep.is_done("force"):
                    log("Remote process finished with state", sheep.state())
                    done = True
                    break
                time.sleep(30)
        except KeyboardInterrupt:
            wait = False
            log("KeyboardInterrupt received...")
        finally:
            if tail_process:
                if wait:
                    # Give some time to tail to do its job.
                    time.sleep(2)
                tail_process.kill()
            if args.attach and not done:
                if sheep.job is not None:
                    log(f"attach is set, killing remote job {sheep.job.job_id}")
                    sheep.job.cancel()
