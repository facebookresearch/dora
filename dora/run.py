# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from functools import partial
import os
from shutil import rmtree
import sys
import typing as tp
import time

from . import git_save
from .executor import start_ddp_workers
from .main import DecoratedMain
from .log import disable_logging, simple_log, red
from .shep import Shepherd

log = partial(simple_log, "Launch:")


def check_job_and_clear(argv: tp.List[str], main: DecoratedMain, clear: bool = False):
    """This will check if an existing job is running and warn,
    unless --clear is passed, in which case we must cancel it.
    """
    shepherd = Shepherd(main, log)
    sheep = shepherd.get_sheep_from_argv(argv)
    if sheep.job is not None:
        shepherd.update()
        if not sheep.is_done():
            job = sheep.job
            log(red(f"Found existing slurm job {job.job_id} with status {job.state}."))
            if clear:
                log("Cancelling the existing job.")
                shepherd.cancel_lazy(sheep.job)
                shepherd.commit()
                time.sleep(3)
            else:
                log(red("PLEASE ABORT NOW UNLESS YOU ARE SURE OF WHAT YOU DO."))
    if clear and sheep.xp.folder.exists():
        log("Removing existing XP folder.")
        try:
            rmtree(sheep.xp.folder)
        except OSError:
            log("Failed to properly remove folder, but things should be okay...")


def run_action(args, main: DecoratedMain):
    xp = main.get_xp(args.argv)
    with ExitStack() as stack:
        if args.git_save and '_DORA_GIT_SAVE_DONE' not in os.environ:
            os.environ['_DORA_GIT_SAVE_DONE'] = '1'
            clone = git_save.get_new_clone(main)
            git_save.assign_clone(xp, clone)
            stack.enter_context(git_save.enter_clone(clone))
            os.execv(sys.executable, [sys.executable, "-m", "dora"] + sys.argv[1:])
        if args.ddp and not os.environ.get('RANK'):
            check_job_and_clear(args.argv, main, args.clear)
            start_ddp_workers(main, args.argv, args.ddp_workers)
        else:
            if 'WORLD_SIZE' not in os.environ:
                check_job_and_clear(args.argv, main, args.clear)
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = '1'
            sys.argv[1:] = args.argv
            disable_logging()  # disable logging to avoid messing up with the wrapped project.
            main()
