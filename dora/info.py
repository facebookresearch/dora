# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The info commands gets the information on a Sheep or XP and can be used
to retrieve the job status, logs etc.
"""
from functools import partial
import json
import os
import shutil
import sys

from .main import DecoratedMain
from .shep import Shepherd
from .log import simple_log, fatal

log = partial(simple_log, "Info:")


def info_action(args, main: DecoratedMain):
    shepherd = Shepherd(main)
    if args.job_id is not None:
        if len(args.argv) > 0:
            fatal("If a job id is provided, you shouldn't pass argv.")
        sheep = shepherd.get_sheep_from_job_id(args.job_id)
        if sheep is None:
            fatal("Could not find any matching sheep")
    else:
        sheep = shepherd.get_sheep_from_argv(args.argv)
    log("Found sheep", sheep)
    log("Folder is", sheep.xp.folder)
    if sheep.log:
        log("Main log is", sheep.log)
    if args.metrics:
        metrics = main.get_xp_history(sheep.xp)
        out = f"Metrics[{len(metrics)}]: "
        if metrics:
            out += json.dumps(metrics[-1])
        log(out)
    if args.cancel:
        if sheep.job is None:
            log("Could not cancel non existing job")
        elif sheep.is_done():
            log("Job is not running")
        else:
            sheep.job.cancel()
    if args.log:
        if sheep.log is None:
            fatal("No log, sheep hasn't been scheduled yet.")
        if not sheep.log.exists():
            fatal(f"Log {sheep.log} does not exist")
        shutil.copyfileobj(open(sheep.log, "r"), sys.stdout, 4096)
    if args.tail:
        if not sheep.log.exists():
            fatal(f"Log {sheep.log} does not exist")
        os.execvp("tail", ["tail", "-n", "200", "-f", sheep.log])
