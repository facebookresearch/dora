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
            log(f"An exception happened while starting workers {exc_value}")
            self.failed = True
        try:
            while self.children and not self.failed:
                for child in list(self.children):
                    try:
                        exitcode = child.wait(0.05)
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

    xp = main.get_xp(argv)
    xp.folder.mkdir(exist_ok=True, parents=True)
    if xp.rendezvous_file.exists():
        xp.rendezvous_file.unlink()
    log(f"Starting {world_size} worker processes for DDP.")
    with ChildrenManager() as manager:
        for rank in range(world_size):
            kwargs = {}
            env = dict(os.environ)
            env['RANK'] = str(rank)
            env['WORLD_SIZE'] = str(world_size)
            args = ["-m", "dora", "-P", package, "run", "--"]
            args += argv
            if rank > 0:
                kwargs['stdin'] = sp.DEVNULL
                kwargs['stdout'] = open(xp.folder / f'worker_{rank}.log', 'w')
                kwargs['stderr'] = sp.STDOUT
            manager.add(
                sp.Popen([sys.executable] + args, env=env, **kwargs))
    sys.exit(int(manager.failed))
