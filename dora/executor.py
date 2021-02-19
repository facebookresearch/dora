# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
"""
Start multiple process locally for DDP.
"""

from functools import partial
import os
import subprocess as sp
import sys

from .log import simple_log, fatal


log = partial(simple_log, "Executor:")


class ChildrenManager:
    def __init__(self):
        self.children = []
        self.failed = False

    def add(self, child):
        child.rank = len(self.children)
        self.children.append(child)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            log("An exception happened while starting workers %r", exc_value)
            self.failed = True
        try:
            while self.children and not self.failed:
                for child in list(self.children):
                    try:
                        exitcode = child.wait(0.1)
                    except sp.TimeoutExpired:
                        continue
                    else:
                        self.children.remove(child)
                        if exitcode:
                            log(f"Worker {child.rank} died, killing all workers")
                            self.failed = True
        except KeyboardInterrupt:
            log("Received keyboard interrupt, trying to kill all workers.")
            self.failed = True
        for child in self.children:
            child.terminate()
        if not self.failed:
            log("All workers completed successfully")


def start_ddp_workers(package, main, argv):
    import torch as th

    world_size = th.cuda.device_count()
    if not world_size:
        fatal(
            "DDP is only available on GPU. Make sure GPUs are properly configured with cuda.")
        sys.exit(1)

    run = main.get_run(argv)
    # If rendezvous file already exist, this can deadlock.
    rendezvous_file = run.folder / run.cfg.ddp.rendezvous_file
    if rendezvous_file.exists():
        rendezvous_file.unlink()
    log(f"Starting {world_size} worker processes for DDP.")
    with ChildrenManager() as manager:
        for rank in range(world_size):
            kwargs = {}
            env = dict(os.environ)
            env['DORA_RANK'] = str(rank)
            env['DORA_WORLD_SIZE'] = str(world_size)
            args = ["-m", "dora", "-P", package]
            args += argv
            if rank > 0:
                kwargs['stdin'] = sp.DEVNULL
                kwargs['stdout'] = open(run.folder / f'worker_{rank}.log', 'w')
                kwargs['stderr'] = sp.STDOUT
            manager.add(
                sp.Popen([sys.executable] + argv, env=env, **kwargs))
    sys.exit(int(manager.failed))
